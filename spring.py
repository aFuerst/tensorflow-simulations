import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
import tensorflow as tf
import sys
import time

m = 1
k = 1
a = -1
totaltime = 10
steps = 10000
log_freq = 1000
num_iterations = steps // log_freq
delta_t = totaltime / steps
prints = []
log_dir = "/home/alfuerst/simulations/checkpoints"
log = "/home/alfuerst/simulations/log.out"

def calc_ke(vel, pos):
    squared = tf.math.square(vel)
    sq_sum = tf.math.reduce_sum(squared, tf.constant([0]))
    mag = tf.math.sqrt(sq_sum)
    x = tf.math.scalar_mul(0.5 * m, tf.math.square(mag))
    return x

def calc_pe(vel, pos):
    return 0.5*k*pos[0]*pos[0]
    
def calc_total_energy(vel, pos):
    return ke(vel) + pe(pos)

def update_force(pos, vel, force, delta_t):
    return tf.math.scalar_mul(-k, pos)

def update_vel(pos, vel, force, delta_t):
    t = tf.math.scalar_mul(0.5 * delta_t / m, force)
    return t + vel

def update_pos(pos, vel, force, delta_t):
    t = tf.math.scalar_mul(delta_t, vel)
    return t + pos

def compute_energies(pos, vel, force, ke, pe, tot, delta_t):
    ke = calc_ke(vel, pos)
    pe = calc_pe(vel, pos)
    tot = ke + pe
    return (ke, pe, tot)

def loop_control(step_target, step, vecs):
    return step < step_target

@tf.function
def loop_execute(pos, vel, force, ke, pe, tot):
    for i in range(log_freq):
        #print(type(vel))
        vel = update_vel(pos, vel, force, delta_t)
        pos = update_pos(pos, vel, force, delta_t)
        force = update_force(pos, vel, force, delta_t)
        vel = update_vel(pos, vel, force, delta_t)
        ke, pe, tot = compute_energies(pos, vel, force, ke, pe, tot, delta_t)
    return (pos, vel, force, ke, pe, tot)

if __name__ == '__main__':
    pos = np.array([a, 0, 0], dtype=np.float64)
    vel = np.array([0, 0, 0], dtype=np.float64)
    force = np.array([-k*a, 0, 0], dtype=np.float64)
    ke = 0.0
    pe = 0.5*k*a*a
    tot = 0.5*k*a
    energies = (ke, pe, tot)

    pos_p = tf.compat.v1.placeholder(dtype=tf.float64, shape=pos.shape)
    vel_p = tf.compat.v1.placeholder(dtype=tf.float64, shape=vel.shape)
    force_p = tf.compat.v1.placeholder(dtype=tf.float64, shape=force.shape)
    ke_p = tf.compat.v1.placeholder(dtype=tf.float64)
    pe_p = tf.compat.v1.placeholder(dtype=tf.float64)
    tot_p = tf.compat.v1.placeholder(dtype=tf.float64)

    with tf.compat.v1.Session() as sess:
        sess.as_default()
        sess.run(tf.compat.v1.global_variables_initializer())
        print("Delta t = {}".format(delta_t))

        # build graph
        start = time.time()
        pos_g, vel_g, force_g, ke_g, pe_g, tot_g = loop_execute(pos_p, vel_p, force_p, ke_p, pe_p, tot_p)
        built = time.time()
        print("build time:", built-start)
        for seg in range(num_iterations):
            feed_dict={pos_p:pos, vel_p:vel, force_p:force, ke_p:ke, pe_p:pe, tot_p:tot}
            pos, vel, force, ke, pe, tot = sess.run([pos_g, vel_g, force_g, ke_g, pe_g, tot_g], feed_dict=feed_dict)
            print((1+seg)*log_freq, pos, vel, force, ke, pe, tot)
        simul = time.time()
        print("sim time:", simul-built)