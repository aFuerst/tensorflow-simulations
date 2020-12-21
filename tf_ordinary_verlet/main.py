import argparse
import math
import os
import shutil
import time
import common
import numpy as np
import tensorflow as tf

import energies
import forces
import tensorflow_manip
from box import Box

tf.compat.v1.logging.set_verbosity('INFO')
tf.compat.v1.disable_eager_execution()


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

def update_pos(pos, vel, edges_half, neg_edges_half, edges, delta_t, ljatom_mass, force, prev_pos, next_pos):
    with tf.name_scope("update_pos"):
        next_pos_val = 2*pos - prev_pos + (force * delta_t*delta_t)/ljatom_mass
        prev_pos_val = pos
        pos_val = next_pos
        next_pos_val = tf.compat.v1.where_v2(next_pos_val > edges_half, next_pos_val - edges, next_pos_val, name="where_edges_half")
        next_pos_val = tf.compat.v1.where_v2(next_pos_val < neg_edges_half, next_pos_val + edges, next_pos_val, name="where_neg_edges_half")
        return (next_pos_val, pos_val, prev_pos_val)

def update_vel(vel, force, delta_t, ljatom_mass_tf, next_pos_p, prev_pos_p):
    with tf.name_scope("update_vel"):
        return (0.5 * (next_pos_p - prev_pos_p))/delta_t

def run_one_iter(vel, pos, force, edges_half, neg_edges_half, edges, delta_t, forces_zeroes_tf, ljatom_mass_tf, ljatom_diameter_tf, prev_pos, next_pos):
    with tf.name_scope("run_one_iter"):
        vel_graph = update_vel(vel, force, delta_t, ljatom_mass_tf, next_pos, prev_pos)
        (next_pos_graph, pos_graph, prev_pos_graph) = update_pos(pos, vel_graph, edges_half, neg_edges_half, edges, delta_t, ljatom_mass_tf, force, prev_pos, next_pos)
        force_graph = forces.lj_force(pos_graph, edges_half, neg_edges_half, edges, forces_zeroes_tf, ljatom_diameter_tf)
        pe_graph = energies.potential_energy(pos, edges_half, neg_edges_half, edges, ljatom_diameter_tf)
        ke_graph = energies.kinetic_energy(vel, ljatom_diameter_tf)
        return vel_graph, pos_graph, force_graph, pe_graph, ke_graph, prev_pos_graph, next_pos_graph

@tf.function
def build_graph(vel_p, pos_p, force_p, edges_half, neg_edges_half, edges, delta_t, log_freq, forces_zeroes_tf, ljatom_mass_tf, ljatom_diameter_tf, prev_pos_p):
    with tf.name_scope("build_graph"):
        v_g, p_g, f_g = vel_p, pos_p, force_p
        pe_g = ke_g = tf.constant(1, dtype=df_type) # placeholders so variable exists in scope for TF
        prev_position_g = prev_pos_p 
        next_position_g = prev_position_g
        for _ in tf.range(log_freq):
            v_g, p_g, f_g, pe_g, ke_g, prev_position_g, next_position_g = run_one_iter(v_g, p_g, f_g, edges_half, neg_edges_half, edges, delta_t, forces_zeroes_tf, ljatom_mass_tf, ljatom_diameter_tf, prev_position_g, next_position_g)
        print("KINETIC 3:"+str(ke_g))
        return v_g, p_g, f_g, pe_g, ke_g, prev_position_g


def save(path, id, velocities, positions, forces, ke):
    np.savetxt(os.path.join(path, "{}-forces".format(id)), forces)
    np.savetxt(os.path.join(path, "{}-velocities".format(id)), velocities)
    np.savetxt(os.path.join(path, "{}-positions".format(id)), positions)
    #np.save(os.path.join(path, "{}-ke".format(id)),ke)


def run_simulation(totaltime=10, steps=10000, log_freq=1000, number_ljatom=108, ljatom_density=0.8442, sess=None, profile=False, xla=True, force_cpu=False, optimizer=False, thread_count=os.cpu_count()):
    tensorflow_manip.toggle_xla(xla)
    tensorflow_manip.manual_optimizer(optimizer)
    config = tensorflow_manip.toggle_cpu(force_cpu, thread_count)
    num_iterations = steps // log_freq
    delta_t = totaltime / steps
    bx = by = bz = pow(number_ljatom/ljatom_density,1.0/3.0) # box edge lengths
    box = Box(bx, by, bz, df_type)
    prev_positions, number_ljatom, masses, diameters = box.fill(number_ljatom, atom_diam=1)
    ljatom_mass_tf = tf.constant(masses, dtype=df_type)
    ljatom_diameter_tf = tf.constant(diameters, dtype=df_type)
    forces = np.zeros(prev_positions.shape, dtype=df_type)
    velocities = np.zeros(prev_positions.shape, dtype=df_type)
    positions = prev_positions + velocities * delta_t + (delta_t*delta_t*0.5*forces)    #positions #np.zeros(positions.shape, dtype=df_type)  
    print("\n positions:"+str(positions[0]))
    print("BOX LENGTH====>"+str(bx))
    edges = box.get_edges_as_tf()
    edges_half = edges / 2.0
    neg_edges_half = tf.negative(edges_half)
    forces_zeroes_tf = tf.constant(np.zeros((number_ljatom, 3)), dtype=df_type)

    position_p = tf.compat.v1.placeholder(dtype=df_type, shape=prev_positions.shape, name="position_placeholder_n")
    prev_position_p = tf.compat.v1.placeholder(dtype=df_type, shape=prev_positions.shape, name="prev_position_placeholder_n")    #placeholder for prev position initialized to 0
    forces_p = tf.compat.v1.placeholder(dtype=df_type, shape=forces.shape, name="forces_placeholder_n")
    velocities_p = tf.compat.v1.placeholder(dtype=df_type, shape=velocities.shape, name="velocities_placeholder_n")

    if sess is None:
        sess = tf.compat.v1.Session(config=config)
        sess.as_default()
    sess.run(tf.compat.v1.global_variables_initializer())
    main_folder = "./outputs"
    if not os.path.exists(main_folder):
        os.mkdir(main_folder)
    subfolder = os.path.join(main_folder, "output-{}-{}-{}-{}-{}".format(thread_count, totaltime, steps, log_freq, number_ljatom))
    if os.path.exists(subfolder):
        shutil.rmtree(subfolder)
    os.mkdir(subfolder)

    v_g, p_g, f_g, pe_g, ke_g, prev_position_g = build_graph(velocities_p, position_p, forces_p, edges_half, neg_edges_half,
         edges, delta_t, log_freq, forces_zeroes_tf, ljatom_mass_tf, ljatom_diameter_tf, prev_position_p)
    #print(ke_g.eval())
    if profile:
        run_options = tf.compat.v1.RunOptions(trace_level=tf.compat.v1.RunOptions.FULL_TRACE)
        run_metadata = tf.compat.v1.RunMetadata()
    else:
        run_options = None
        run_metadata = None
    writer = tf.compat.v1.summary.FileWriter(subfolder)
    writer.add_graph(sess.graph)
    comp_start = time.time()
    totalEnergy = ke_g + pe_g
    save(subfolder, 0, velocities, positions, forces, totalEnergy)
    
    for x in range(num_iterations):
        feed_dict = {position_p:positions, forces_p:forces, velocities_p:velocities, prev_position_p:prev_positions, }
        velocities, positions, forces, pe, ke, prev_positions = sess.run([v_g, p_g, f_g, pe_g, ke_g, prev_position_g], feed_dict=feed_dict, options=run_options, run_metadata=run_metadata)
        #print("Kinetic energy 5:"+str(ke_g.eval()))    
        #tf.compat.v1.logging.info('@vinita ' + str(ke))   
        tf.compat.v1.logging.info('KE:'+str(ke))
        tf.compat.v1.logging.info('PE:'+str(pe))
        tf.compat.v1.logging.info('total energy:'+str(ke+pe))
        if profile:
            from tensorflow.python.client import timeline
            tl = timeline.Timeline(run_metadata.step_stats)
            ctf = tl.generate_chrome_trace_format()
            writer.add_run_metadata(run_metadata, 'step%d' % x)
            with open(os.path.join(subfolder, "{}-timeline.json".format((1+x)*log_freq)), 'w') as f:
                f.write(ctf)
        # with tf.session() as sess:
        #    print("Kinetic energy 5:"+ke_g.eval())
        save(subfolder, (1+x)*log_freq, velocities, positions, forces, ke)
    return time.time() - comp_start

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c',"--cpu", action="store_true")
    parser.add_argument('-x', "--xla", action="store_true")
    parser.add_argument('-r', "--prof", action="store_true")
    parser.add_argument('-o', "--opt", action="store_true")
    parser.add_argument('-p', "--parts", action="store", default=108, type=int)
    parser.add_argument('-s', "--steps", action="store", default=10000, type=int)
    parser.add_argument('-t', "--time", action="store", default=10, type=int)
    parser.add_argument('-l', "--log", action="store", default=1000, type=int)
    parser.add_argument("--threads", action="store", default=os.cpu_count(), type=int)
    args = parser.parse_args()
    comp = run_simulation(profile=args.prof, xla=args.xla, force_cpu=args.cpu, number_ljatom=args.parts, steps=args.steps, log_freq=args.log, totaltime=args.time, optimizer=args.opt, thread_count=args.threads)
    print(comp)
