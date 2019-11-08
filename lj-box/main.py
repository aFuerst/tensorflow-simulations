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
positions = tf.Variable(box.fill(number_ljatom, ljatom_diameter))
forces = tf.Variable(np.asarray([[1,2,3]], dtype=np.float64))
velocities = tf.Variable(np.zeros(positions.shape))
edges = tf.Variable(np.array([bx,by,bz]))

with tf.compat.v1.Session() as sess:
    sess.as_default()
    writer = tf.compat.v1.summary.FileWriter('./test', sess.graph)
    sess.run(tf.compat.v1.global_variables_initializer())
    print("Delta t = {}".format(delta_t))
    # for seg in range(num_iterations):
    run_options = tf.compat.v1.RunOptions(trace_level=tf.compat.v1.RunOptions.FULL_TRACE)
    run_metadata = tf.compat.v1.RunMetadata()
    for x in range(5):
        a = time.time()
        for i in range(100):
            tf.compat.v1.summary.scalar('accuracy', positions)
            velocities = velocities + (forces * (0.5*delta_t/ljatom_mass))
            positions = positions + (velocities * delta_t)
            positions = positions % edges
        # merged = tf.compat.v1.summary.merge_all()
        writer.add_graph(sess.graph)
        positions, velocities = sess.run([positions, velocities]) #, options=run_options,  run_metadata=run_metadata)
        # writer.add_run_metadata(run_metadata, x)
        # writer.add_summary(summaries, x)
        b = time.time()
        print(b-a)
    writer.close()