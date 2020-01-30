import numpy as np
import tensorflow as tf
from box import Box
import energies, forces, common
import time, argparse, shutil, math, sys, os

pi = 3.141593
kB = 1.38e-23
mol = 6.0e-23
unit_len = 0.4505e-9
unit_energy = 119.8 * kB # Joules; unit of energy is these many Joules (typical interaction strength)
unit_mass = 0.03994 / mol # kg; unit of mass (mass of a typical atom)
unit_time = math.sqrt(unit_mass * unit_len * unit_len / unit_energy) # Unit of time
unit_temp = unit_energy/kB # unit of temperature
df_type=np.float64

temperature = 1.0
dcut = 1 # cutoff distance for potential in reduced units

def update_pos(pos, vel, edges_half, neg_edges_half, edges, delta_t):
    with tf.name_scope("update_pos"):
        pos_graph = pos + (vel * delta_t)
        pos_graph = tf.compat.v1.where_v2(pos_graph > edges_half, pos_graph - edges, pos_graph)
        return tf.compat.v1.where_v2(pos_graph < neg_edges_half, pos_graph + edges, pos_graph)

def update_vel(vel, force, delta_t, ljatom_mass_tf):
    with tf.name_scope("update_vel"):
        return vel + (force * (0.5*delta_t/ljatom_mass_tf))

def run_one_iter(vel, pos, force, edges_half, neg_edges_half, edges, delta_t, forces_zeroes_tf, ljatom_mass_tf, ljatom_diameter_tf):
    with tf.name_scope("run_one_iter"):
        vel_graph = update_vel(vel, force, delta_t, ljatom_mass_tf)
        pos_graph = update_pos(pos, vel_graph, edges_half, neg_edges_half, edges, delta_t)
        force_graph = forces.lj_force(pos_graph, edges_half, neg_edges_half, edges, forces_zeroes_tf, ljatom_diameter_tf)
        pe_graph = energies.potential_energy(pos, edges_half, neg_edges_half, edges, ljatom_diameter_tf)
        ke_graph = energies.kinetic_energy(vel, ljatom_diameter_tf)
        vel_graph = update_vel(vel, force, delta_t, ljatom_mass_tf)
        return vel_graph, pos_graph, force_graph, pe_graph, ke_graph

@tf.function
def build_graph(vel_p, pos_p, force_p, edges_half, neg_edges_half, edges, delta_t, log_freq, forces_zeroes_tf, ljatom_mass_tf, ljatom_diameter_tf):
    with tf.name_scope("build_graph"):
        v_g, p_g, f_g = vel_p, pos_p, force_p
        pe_g = ke_g = tf.constant(1, dtype=df_type) # placeholders so variable exists in scope for TF
        for _ in tf.range(log_freq):
            v_g, p_g, f_g, pe_g, ke_g = run_one_iter(v_g, p_g, f_g, edges_half, neg_edges_half, edges, delta_t, forces_zeroes_tf, ljatom_mass_tf, ljatom_diameter_tf)
        return v_g, p_g, f_g, pe_g, ke_g

def save(path, id, velocities, positions, forces):
    np.savetxt(os.path.join(path, "{}-forces".format(id)), forces)
    np.savetxt(os.path.join(path, "{}-velocities".format(id)), velocities)
    np.savetxt(os.path.join(path, "{}-positions".format(id)), positions)

def toggle_xla(xla):
    if xla:
        tf.config.optimizer.set_jit(xla)

def toggle_cpu(cpu, thread_count):
    if cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        tf.config.threading.set_inter_op_parallelism_threads(thread_count)
        tf.config.threading.set_intra_op_parallelism_threads(thread_count)
        return tf.compat.v1.ConfigProto(intra_op_parallelism_threads=thread_count, inter_op_parallelism_threads=thread_count)
    return None

def manual_optimizer(optimizer):
    if optimizer:
        tf.config.optimizer.set_experimental_options({'constant_folding': True, "layout_optimizer": True, "shape_optimization":True, "remapping":True, "arithmetic_optimization":True, "dependency_optimization":True, "loop_optimization":True, "function_optimization":True, "debug_stripper":True, "scoped_allocator_optimization":True, "implementation_selector":True, "auto_mixed_precision":True, "pin_to_host_optimization":True})

def run_simulation(totaltime=10, steps=10000, log_freq=1000, number_ljatom=108, ljatom_density=0.8442, sess=None, profile=False, xla=True, force_cpu=False, optimizer=False, thread_count=os.cpu_count()):
    toggle_xla(xla)
    manual_optimizer(optimizer)
    config = toggle_cpu(force_cpu, thread_count)
    num_iterations = steps // log_freq
    delta_t = totaltime / steps
    bx = by = bz = pow(number_ljatom/ljatom_density,1.0/3.0) # box edge lengths
    box = Box(bx, by, bz, df_type)
    positions, number_ljatom, masses, diameters = box.fill(number_ljatom, atom_diam=1)
    ljatom_mass_tf = tf.constant(masses, dtype=df_type)
    ljatom_diameter_tf = tf.constant(diameters, dtype=df_type)
    forces = np.zeros(positions.shape, dtype=df_type)
    velocities = np.zeros(positions.shape, dtype=df_type)
    edges = box.get_edges_as_tf()
    edges_half = edges / 2.0
    neg_edges_half = tf.negative(edges_half)
    forces_zeroes_tf = tf.constant(np.zeros((number_ljatom, 3)), dtype=df_type)

    position_p = tf.compat.v1.placeholder(dtype=df_type, shape=positions.shape, name="position_placeholder_n")
    forces_p = tf.compat.v1.placeholder(dtype=df_type, shape=forces.shape, name="forces_placeholder_n")
    velocities_p = tf.compat.v1.placeholder(dtype=df_type, shape=velocities.shape, name="velocities_placeholder_n")

    if sess is None:
        sess = tf.compat.v1.Session(config=config)
        sess.as_default()
    sess.run(tf.compat.v1.global_variables_initializer())
    main_folder = "./outputs"
    if not os.path.exists(main_folder):
        os.mkdir(main_folder)
    subfolder = os.path.join(main_folder, "output-{}-{}-{}-{}".format(totaltime, steps, log_freq, number_ljatom))
    if os.path.exists(subfolder):
        shutil.rmtree(subfolder)
    os.mkdir(subfolder)

    v_g, p_g, f_g, pe_g, ke_g = build_graph(velocities_p, position_p, forces_p, edges_half, neg_edges_half,\
         edges, delta_t, log_freq, forces_zeroes_tf, ljatom_mass_tf, ljatom_diameter_tf)

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
            writer.add_run_metadata(run_metadata, 'step%d' % x)
            with open(os.path.join(subfolder, "{}-timeline.json".format((1+x)*log_freq)), 'w') as f:
                f.write(ctf)
        save(subfolder, (1+x)*log_freq, velocities, positions, forces)
    return time.time() - comp_start
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c',"--cpu", action="store_true")
    parser.add_argument('-x', "--xla", action="store_true")
    parser.add_argument('-r', "--prof", action="store_true")
    parser.add_argument('-o', "--opt", action="store_true")
    parser.add_argument('-p', "--parts", action="store", default=108)
    parser.add_argument('-s', "--steps", action="store", default=10000)
    parser.add_argument('-t', "--time", action="store", default=10)
    parser.add_argument('-l', "--log", action="store", default=1000)
    parser.add_argument("--threads", action="store", default=os.cpu_count())
    args = parser.parse_args()
    comp = run_simulation(profile=args.prof, xla=args.xla, force_cpu=args.cpu, number_ljatom=int(args.parts), steps=int(args.steps), log_freq=int(args.log), totaltime=int(args.time), optimizer=args.opt, thread_count=int(args.threads))
    print(comp)