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
number_ljatom = 108 # total number of particles in your system
ljatom_diameter	= 1
ljatom_mass = 1
bx = by = bz  = pow(number_ljatom/ljatom_density,1.0/3.0) # box edge lengths
temperature = 1
dcut = 1 # cutoff distance for potential in reduced units

totaltime = 10
steps = 10000
log_freq = 1000
num_iterations = steps // log_freq
delta_t = totaltime / steps

box = box.Box(bx, by, bz)
positions = box.fill(number_ljatom, ljatom_diameter)
forces = np.asarray([[1,2,3]], dtype=np.float64)
velocities = np.zeros(positions.shape, dtype=np.float64)
edges = box.get_edges_as_tf()

position_p = tf.compat.v1.placeholder(dtype=tf.float64, shape=positions.shape, name="position_placeholder_n")
forces_p = tf.compat.v1.placeholder(dtype=tf.float64, shape=forces.shape, name="forces_placeholder_n")
velocities_p = tf.compat.v1.placeholder(dtype=tf.float64, shape=velocities.shape, name="velocities_placeholder_n")

@tf.function
def run_one_iter(vel, pos, force):
    vel_graph = vel + (force * (0.5*delta_t/ljatom_mass))
    pos_graph = pos + (vel_graph * delta_t)
    pos_graph = pos_graph % edges
    # update forces
    # compute energies
    vel_graph = vel_graph + (force * (0.5*delta_t/ljatom_mass))
    return vel_graph, pos_graph, force

def build_graph(num_iterations):
    # build graph
    v_g, p_g, f_g = run_one_iter(velocities_p, position_p, forces_p)
    for i in range(num_iterations-1):
        run_one_iter(v_g, p_g, f_g)
    return v_g, p_g, f_g

with tf.compat.v1.Session() as sess:
    sess.as_default()
    sess.run(tf.compat.v1.global_variables_initializer())
    writer = tf.compat.v1.summary.FileWriter("./test", sess.graph)
    v_g, p_g, f_g = build_graph(100)
    writer.add_graph(sess.graph)
    writer.close()

    for x in range(10):
        a = time.time()
        feed_dict = {position_p:positions, forces_p:forces, velocities_p:velocities}
        velocities, positions, forces = sess.run([v_g, p_g, f_g], feed_dict=feed_dict)
        # print(velocities, positions, forces)
        b = time.time()
        print(b-a)