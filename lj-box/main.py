import os
import numpy as np
import tensorflow as tf
import sys
import math
from box import Box
import time

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

def calc_force(pos, edges_half, neg_edges_half, edges, forces_zeroes_tf):
    # TODO: remove hack of passing 'pos' into single_force. just uses scoping rules to 'pass in'
    def single_force(atom_pos):
        dist = pos - atom_pos
        dist = tf.compat.v1.where_v2(dist > edges_half, dist - edges, dist)
        dist = tf.compat.v1.where_v2(dist < neg_edges_half, dist + edges, dist)
        magnitude = tf.math.sqrt(tf.math.reduce_sum(tf.math.pow(dist,2.0), axis=1, keepdims=True))
        twelve = tf.math.pow(ljatom_diameter_tf, 12.0) / tf.math.pow(magnitude, 12.0)
        six = tf.math.pow(ljatom_diameter_tf, 6.0) / tf.math.pow(magnitude, 6.0)
        mag_squared = 1.0 / tf.math.pow(magnitude,2)
        
        slice_forces = dist * (48.0 * 1.0 * (((twelve - 0.5 * six) * mag_squared)))
        # handle case when pos - atom_pos == 0, causing inf and nan to appear in that position 
        # can't see a way to remove that case in all above computations, easier to do it all at once at the end
        filter = tf.math.logical_or(tf.math.is_nan(slice_forces), magnitude < (ljatom_diameter_tf*2.5))
        filtered = tf.compat.v1.where_v2(filter, forces_zeroes_tf, slice_forces)
        return tf.math.reduce_sum(filtered, axis=0)
    return tf.compat.v1.vectorized_map(fn=single_force, elems=pos)

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
    # TODO: compute energies
    vel_graph = update_vel(vel, force, delta_t)
    return vel_graph, pos_graph, force_graph

@tf.function
def build_graph(vel_p, pos_p, force_p, edges_half, neg_edges_half, edges, delta_t, log_freq, forces_zeroes_tf):
    v_g, p_g, f_g = vel_p, pos_p, force_p
    for _ in tf.range(log_freq):
        v_g, p_g, f_g = run_one_iter(v_g, p_g, f_g, edges_half, neg_edges_half, edges, delta_t, forces_zeroes_tf)
    return v_g, p_g, f_g

def save(timestamp, id, velocities, positions, forces):
    # TODO: better/faster way to do? (async?)
    np.savetxt("./outputs/output-{0}/{1}-forces".format(timestamp, id), forces)
    np.savetxt("./outputs/output-{0}/{1}-velocities".format(timestamp, id), velocities)
    np.savetxt("./outputs/output-{0}/{1}-positions".format(timestamp, id), positions)

def run_simulation(totaltime=10, steps=10000, log_freq=1000, number_ljatom=108, ljatom_density = 0.8442, sess=None):
    log_freq_tf = tf.constant(log_freq)
    num_iterations = steps // log_freq
    delta_t = totaltime / steps
    bx = by = bz  = pow(number_ljatom/ljatom_density,1.0/3.0) # box edge lengths
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
    beg = time.time()
    if not os.path.exists("./outputs"):
        os.mkdir("./outputs")
    os.mkdir("./outputs/output-{}".format(beg))
    
    v_g, p_g, f_g = build_graph(velocities_p, position_p, forces_p, edges_half, neg_edges_half, edges, delta_t, log_freq, forces_zeroes_tf)
    built = time.time()
    # print("Graph build time:", built - beg)
    
    writer = tf.compat.v1.summary.FileWriter("./outputs/output-{}".format(beg))
    writer.add_graph(sess.graph)
    writer.close()
    disk = time.time()
    save(beg, 0, velocities, positions, forces)
    # print("Graph saved time:", disk - built)
    timings = []
    for x in range(num_iterations):
        a = time.time()
        feed_dict = {position_p:positions, forces_p:forces, velocities_p:velocities}
        velocities, positions, forces = sess.run([v_g, p_g, f_g], feed_dict=feed_dict)
        save(beg, (1+x)*log_freq, velocities, positions, forces)
        timings.append((x, time.time()-a))
        # print("Iteration done: ", time.time()-a)
    comp = time.time() - disk
    print("Computation time:", comp)
    return timings, comp

if __name__ == "__main__":
    run_simulation()