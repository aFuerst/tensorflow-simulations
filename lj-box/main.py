import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
import tensorflow as tf
import sys
import math
import box
import time

pi = 3.141593
kB = 1.38e-23
mol = 6.0e-23
unit_len = 0.4505e-9
unit_energy = 119.8 * kB # Joules; unit of energy is these many Joules (typical interaction strength)
unit_mass = 0.03994 / mol # kg; unit of mass (mass of a typical atom)
unit_time = math.sqrt(unit_mass * unit_len * unit_len / unit_energy) # Unit of time
unit_temp = unit_energy/kB # unit of temperature

ljatom_density = 0.8442 # this is taken in as mass density (in reduced units)
number_ljatom = 108 # total number of particles in your system (108)
ljatom_diameter	= 1.0
ljatom_diameter_tf = tf.constant(ljatom_diameter, dtype=tf.float64)
ljatom_mass = 1.0
ljatom_mass_tf = tf.constant(ljatom_mass, dtype=tf.float64)
bx = by = bz  = pow(number_ljatom/ljatom_density,1.0/3.0) # box edge lengths
temperature = 1.0
dcut = 1 # cutoff distance for potential in reduced units

totaltime = 10
steps = 10000
log_freq = 1000
log_freq_tf = tf.constant(log_freq)
num_iterations = steps // log_freq
num_iterations_tf = tf.constant(num_iterations)
delta_t = totaltime / steps
delta_t_tf = tf.constant(delta_t, dtype=np.float64)
print("Delta t:", delta_t)
print("Edges = {}, {}, {}".format(bx, by, bz))

box = box.Box(bx, by, bz)
positions, number_ljatom = box.fill(number_ljatom, ljatom_diameter)
number_ljatom_tf = tf.constant(number_ljatom, dtype=tf.int32)
forces = np.zeros(positions.shape, dtype=np.float64)
velocities = np.zeros(positions.shape, dtype=np.float64)
edges = box.get_edges_as_tf()
edges_half = box.get_edges_as_tf() / 2.0
forces_zeroes_tf = tf.constant(np.zeros((number_ljatom, 3)), dtype=tf.float64)
forces_mag_zeroes_tf = tf.constant(np.zeros((number_ljatom, 1)), dtype=tf.float64)

position_p = tf.compat.v1.placeholder(dtype=tf.float64, shape=positions.shape, name="position_placeholder_n")
forces_p = tf.compat.v1.placeholder(dtype=tf.float64, shape=forces.shape, name="forces_placeholder_n")
velocities_p = tf.compat.v1.placeholder(dtype=tf.float64, shape=velocities.shape, name="velocities_placeholder_n")

@tf.function
def calc_force(pos):
    # TODO: remove hack of passing 'pos' into single_force. just uses scoping rules to 'pass in'
    def single_force(atom_pos):
        dist = pos - atom_pos
        dist = tf.where(tf.math.abs(dist) > edges_half, dist, dist - edges)
        dist = tf.where(tf.math.abs(dist) < edges_half, dist, dist + edges)
        magnitude = tf.math.sqrt(tf.math.reduce_sum(dist**2.0, axis=1, keepdims=True))
        twelve = ljatom_diameter_tf ** 12.0 / magnitude **12.0
        six = ljatom_diameter_tf ** 6.0 / magnitude ** 6.0
        mag_squared = 1.0 / (magnitude**2)
        
        slice_forces = dist * (48.0 * 1.0 * ((twelve - 0.5 * six * mag_squared)))
        # handle case when pos - atom_pos == 0, causing inf and nan to appear in that position 
        # can't see a way to remove that case in all above computations, easier to do it all at once at the end
        filter = tf.math.logical_or(tf.math.is_inf(slice_forces), tf.math.is_nan(slice_forces))
        filter = tf.math.logical_or(filter, magnitude < (ljatom_diameter_tf*2.5))
        filtered = tf.where(filter, forces_zeroes_tf, slice_forces)
        # slice_forces = tf.debugging.check_numerics(slice_forces, "slice_forces")
  
        return tf.math.reduce_sum(filtered, axis=0)

    return tf.map_fn(fn=single_force, elems=pos, parallel_iterations=number_ljatom)

def run_one_iter(vel, pos, force):
    vel_graph = vel + (force * (0.5*delta_t_tf/ljatom_mass_tf))
    pos_graph = pos + (vel_graph * delta_t)
    pos_graph = tf.where(tf.math.abs(pos_graph) > edges_half, pos_graph, pos_graph - edges)
    pos_graph = tf.where(tf.math.abs(pos_graph) < edges_half, pos_graph, pos_graph + edges)
    force_graph = calc_force(pos_graph)
    # TODO: compute energies
    vel_graph = vel_graph + (force_graph * (0.5*delta_t_tf/ljatom_mass_tf))
    return vel_graph, pos_graph, force_graph

@tf.function
def build_graph(vel_p, pos_p, force_p):
    v_g, p_g, f_g = vel_p, pos_p, force_p
    for _ in tf.range(log_freq_tf):
        v_g, p_g, f_g = run_one_iter(v_g, p_g, f_g)
    return v_g, p_g, f_g

def save(timestamp, id, velocities, positions, forces):
    # TODO: better/faster way to do? (async?)
    np.savetxt("./outputs/output-{0}/{1}-forces".format(timestamp, id), forces)
    np.savetxt("./outputs/output-{0}/{1}-velocities".format(timestamp, id), velocities)
    np.savetxt("./outputs/output-{0}/{1}-positions".format(timestamp, id), positions)

with tf.compat.v1.Session() as sess:
    sess.as_default()
    sess.run(tf.compat.v1.global_variables_initializer())
    beg = time.time()
    os.mkdir("./outputs/output-{}".format(beg))
    
    feed_dict = {position_p:positions, forces_p:forces, velocities_p:velocities}
    f_g = calc_force(position_p) # initialize forces
    forces = sess.run(f_g, feed_dict=feed_dict)
    if np.any(np.isnan(forces)): # TODO: remove when everything else works
        raise AssertionError("Got nan in initial forces:\n {}".format(forces))

    v_g, p_g, f_g = build_graph(velocities_p, position_p, forces_p)
    built = time.time()
    print("Graph build time:", built - beg)
    
    writer = tf.compat.v1.summary.FileWriter("./outputs/output-{}".format(beg))
    writer.add_graph(sess.graph)
    writer.close()
    disk = time.time()
    save(beg, 0, velocities, positions, forces)
    print("Graph saved time:", disk - built)
    
    for x in range(num_iterations):
        feed_dict = {position_p:positions, forces_p:forces, velocities_p:velocities}
        velocities, positions, forces = sess.run([v_g, p_g, f_g], feed_dict=feed_dict)
        save(beg, (1+x)*log_freq, velocities, positions, forces)
        print("Iteration done:", x)
    print("Computation time:", time.time() - disk)