import numpy as np
import tensorflow as tf
from box import Box
import time, argparse, shutil, math, sys, os

pi = 3.141593
kB = 1.38e-23
mol = 6.0e-23
unit_len = 0.4505e-9
unit_energy = 119.8 * kB # Joules; unit of energy is these many Joules (typical interaction strength)
unit_mass = 0.03994 / mol # kg; unit of mass (mass of a typical atom)
unit_time = math.sqrt(unit_mass * unit_len * unit_len / unit_energy) # Unit of time
unit_temp = unit_energy/kB # unit of temperature

ljatom_diameter	= 1.0
ljatom_diameter_tf = tf.constant(ljatom_diameter, dtype=tf.float64)
ljatom_mass = 1.0
ljatom_mass_tf = tf.constant(ljatom_mass, dtype=tf.float64)
temperature = 1.0
dcut = 1 # cutoff distance for potential in reduced units

def calc_magnitude(tensor):
    return tf.math.sqrt(tf.math.reduce_sum(tf.math.pow(tensor,2.0), axis=1, keepdims=True))

def calc_force(pos, edges_half, neg_edges_half, edges, forces_zeroes_tf):
    distances = tf.compat.v1.vectorized_map(fn=lambda atom_pos: pos - atom_pos, elems=pos)
    distances = tf.compat.v1.where_v2(distances > edges_half, distances - edges, distances)
    distances = tf.compat.v1.where_v2(distances < neg_edges_half, distances + edges, distances)
    magnitude = calc_magnitude(distances)
    twelve = tf.math.pow(ljatom_diameter_tf, 12.0) / tf.math.pow(magnitude, 12.0)
    six = tf.math.pow(ljatom_diameter_tf, 6.0) / tf.math.pow(magnitude, 6.0)
    mag_squared = 1.0 / tf.math.pow(magnitude,2)
    slice_forces = distances * (48.0 * 1.0 * (((twelve - 0.5 * six) * mag_squared)))
    # handle case distances pos - atom_pos == 0, causing inf and nan to appear in that position
    # can't see a way to remove that case in all above computations, easier to do it all at once at the end
    filter = tf.math.logical_or(tf.math.is_nan(slice_forces), magnitude < (ljatom_diameter_tf*2.5))
    filtered = tf.compat.v1.where_v2(filter, forces_zeroes_tf, slice_forces)
    forces =  tf.math.reduce_sum(filtered, axis=0)
    return forces

def update_pos(pos, vel, edges_half, neg_edges_half, edges, delta_t):
    pos_graph = pos + (vel * delta_t)
    pos_graph = tf.compat.v1.where_v2(pos_graph > edges_half, pos_graph - edges, pos_graph)
    return tf.compat.v1.where_v2(pos_graph < neg_edges_half, pos_graph + edges, pos_graph)

def update_vel(vel, force, delta_t):
    return vel + (force * (0.5*delta_t/ljatom_mass_tf))

def run_one_iter(vel, pos, force, edges_half, neg_edges_half, edges, delta_t, forces_zeroes_tf):
    vel_graph = update_vel(vel, force, delta_t)
    pos_graph = update_pos(pos, vel_graph, edges_half, neg_edges_half, edges, delta_t)
    force_graph = calc_force(pos_graph, edges_half, neg_edges_half, edges, forces_zeroes_tf)
    pe_graph = calc_pe(pos, edges_half, neg_edges_half, edges, forces_zeroes_tf)
    ke_graph = calc_ke(vel)
    vel_graph = update_vel(vel, force, delta_t)
    return vel_graph, pos_graph, force_graph, pe_graph, ke_graph

def calc_ke(vel):
    magnitude = calc_magnitude(vel)
    return tf.reduce_sum(0.5 * 1 * magnitude * magnitude)

def calc_pe(pos, edges_half, neg_edges_half, edges, forces_zeroes_tf):
    dcut = tf.constant(2.5, dtype=tf.float64)
    elj = tf.constant(1.0, dtype=tf.float64)
    particle_diam = tf.constant(1.0, dtype=tf.float64)
    dcut_6 = tf.pow(dcut, 6)
    dcut_12 = tf.pow(dcut, 12)
    energy_shift = 4.0*elj*(1.0/dcut_12 - 1.0/dcut_6)
    distances = tf.compat.v1.vectorized_map(fn=lambda atom_pos: pos - atom_pos, elems=pos)
    distances = tf.compat.v1.where_v2(distances > edges_half, distances - edges, distances)
    distances = tf.compat.v1.where_v2(distances < neg_edges_half, distances + edges, distances)
    magnitude = calc_magnitude(distances)
    d_6 = tf.pow(particle_diam, 6)
    r_6 = tf.pow(magnitude, 6)
    ljpair = 4.0 * elj * (d_6 / r_6) * ( ( d_6 / r_6 ) - 1.0 ) - energy_shift
    ret = tf.reduce_sum(ljpair) / 2.0
    return ret

@tf.function
def build_graph(vel_p, pos_p, force_p, edges_half, neg_edges_half, edges, delta_t, log_freq, forces_zeroes_tf):
    v_g, p_g, f_g = vel_p, pos_p, force_p
    pe_g = ke_g = tf.constant(1, dtype=tf.float64) # placeholders so variable exists in scope for TF
    for _ in tf.range(log_freq):
        v_g, p_g, f_g, pe_g, ke_g = run_one_iter(v_g, p_g, f_g, edges_half, neg_edges_half, edges, delta_t, forces_zeroes_tf)
    return v_g, p_g, f_g, pe_g, ke_g

def save(path, id, velocities, positions, forces):
    np.savetxt(os.path.join(path, "{}-forces".format(id)), forces)
    np.savetxt(os.path.join(path, "{}-velocities".format(id)), velocities)
    np.savetxt(os.path.join(path, "{}-positions".format(id)), positions)

def toggle_xla(xla):
    if xla:
        tf.config.optimizer.set_jit(xla)

def toggle_cpu(cpu):
    if cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def run_simulation(totaltime=10, steps=10000, log_freq=1000, number_ljatom=108, ljatom_density=0.8442, sess=None, profile=False, xla=True, force_cpu=False):
    toggle_xla(xla)
    toggle_cpu(force_cpu)
    num_iterations = steps // log_freq
    delta_t = totaltime / steps
    bx = by = bz = pow(number_ljatom/ljatom_density,1.0/3.0) # box edge lengths
    box = Box(bx, by, bz)
    positions, number_ljatom = box.fill(number_ljatom, ljatom_diameter)
    forces = np.zeros(positions.shape, dtype=np.float64)
    velocities = np.zeros(positions.shape, dtype=np.float64)
    edges = box.get_edges_as_tf()
    edges_half = box.get_edges_as_tf() / 2.0
    neg_edges_half = tf.negative(edges_half)
    forces_zeroes_tf = tf.constant(np.zeros((number_ljatom, 3)), dtype=tf.float64)

    position_p = tf.compat.v1.placeholder(dtype=tf.float64, shape=positions.shape, name="position_placeholder_n")
    forces_p = tf.compat.v1.placeholder(dtype=tf.float64, shape=forces.shape, name="forces_placeholder_n")
    velocities_p = tf.compat.v1.placeholder(dtype=tf.float64, shape=velocities.shape, name="velocities_placeholder_n")

    if sess is None:
        sess = tf.compat.v1.Session()
        sess.as_default()
    sess.run(tf.compat.v1.global_variables_initializer())
    main_folder = "./outputs"
    if not os.path.exists(main_folder):
        os.mkdir(main_folder)
    subfolder = os.path.join(main_folder, "output-{}-{}-{}-{}".format(totaltime, steps, log_freq, number_ljatom))
    if os.path.exists(subfolder):
        shutil.rmtree(subfolder)
    os.mkdir(subfolder)

    v_g, p_g, f_g, pe_g, ke_g = build_graph(velocities_p, position_p, forces_p, edges_half, neg_edges_half, edges, delta_t, log_freq, forces_zeroes_tf)

    if profile:
        run_options = tf.compat.v1.RunOptions(trace_level=tf.compat.v1.RunOptions.FULL_TRACE)
        run_metadata = tf.compat.v1.RunMetadata()
    else:
        run_options = None
        run_metadata = None
    writer = tf.compat.v1.summary.FileWriter(subfolder)
    writer.add_graph(sess.graph)
    comp_start = time.time()
    save(subfolder, 0, velocities, positions, forces)
    for x in range(num_iterations):
        feed_dict = {position_p:positions, forces_p:forces, velocities_p:velocities}
        velocities, positions, forces, pe, ke = sess.run([v_g, p_g, f_g, pe_g, ke_g], feed_dict=feed_dict, options=run_options, run_metadata=run_metadata)
        if profile:
            from tensorflow.python.client import timeline
            tl = timeline.Timeline(run_metadata.step_stats)
            ctf = tl.generate_chrome_trace_format()
            with open(os.path.join(subfolder, "{}-timeline.json".format((1+x)*log_freq)), 'w') as f:
                f.write(ctf)
        save(subfolder, (1+x)*log_freq, velocities, positions, forces)
    return time.time() - comp_start
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c',"--cpu", action="store_true")
    parser.add_argument('-x', "--xla", action="store_true")
    parser.add_argument('-r', "--prof", action="store_true")
    parser.add_argument('-p', "--parts", action="store", default=108)
    parser.add_argument('-s', "--steps", action="store", default=10000)
    parser.add_argument('-t', "--time", action="store", default=10)
    parser.add_argument('-l', "--log", action="store", default=1000)
    args = parser.parse_args()
    # print(args)
    comp = run_simulation(profile=args.prof, xla=args.xla, force_cpu=args.cpu, number_ljatom=int(args.parts), steps=int(args.steps), log_freq=int(args.log), totaltime=int(args.time))
    print(comp)