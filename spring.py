import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
import tensorflow as tf
import sys

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

def compute_energies(pos, vel, force, energies, delta_t):
    ke, pe, tot = energies
    ke = calc_ke(vel, pos)
    pe = calc_pe(vel, pos)
    tot = ke + pe
    return (ke, pe, tot)

def loop_control(step_target, step, vecs):
    return step < step_target

@tf.function
def loop_execute(vecs):
    pos, vel, force, energies = vecs
    for i in range(log_freq):
        #print(type(vel))
        vel = update_vel(pos, vel, force, delta_t)
        pos = update_pos(pos, vel, force, delta_t)
        force = update_force(pos, vel, force, delta_t)
        vel = update_vel(pos, vel, force, delta_t)
        energies = compute_energies(pos, vel, force, energies, delta_t)
    return (pos, vel, force, energies)

if __name__ == '__main__':
    pos = tf.Variable(np.array([a, 0, 0], dtype=np.float64), name="position")
    vel = tf.Variable(np.array([0, 0, 0], dtype=np.float64), name="velocity")
    force = tf.Variable(np.array([-k*a, 0, 0], dtype=np.float64), name="force")
    ke = tf.Variable(0, dtype=np.float64, name="kinetic")
    pe = tf.Variable(0.5*k*a*a, dtype=np.float64, name="potential")
    tot = tf.Variable(0.5*k*a, dtype=np.float64, name="total")
    energies = (ke, pe, tot)
    
    with tf.compat.v1.Session() as sess:
        sess.as_default()
        sess.run(tf.compat.v1.global_variables_initializer())
        print("Delta t = {}".format(delta_t))
        out = (pos, vel, force, energies)
        for seg in range(num_iterations):
            out = loop_execute(out)
            out = sess.run(out)
            print((1+seg)*log_freq, out)