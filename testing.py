import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
import tensorflow as tf
# import shutil
import sys
# shutil.rmtree("checkpoints")
# os.mkdir("checkpoints")
# force CPU for testing
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

m = 1
k = 1
a = -1
totaltime = 10
steps = 10000
log_freq = 100
num_iterations = steps // log_freq
delta_t = totaltime / steps
prints = []
log_dir = "/home/alfuerst/simulations/checkpoints"
log = "/home/alfuerst/simulations/log.out"

def my_print(*args):
    op = tf.print(args, output_stream=log)
    prints.append(op)
    return op

# @tf.function
def calc_ke(vel, pos):
    squared = tf.math.square(vel)
    sq_sum = tf.math.reduce_sum(squared, tf.constant([0]))
    mag = tf.math.sqrt(sq_sum)
    x = tf.math.scalar_mul(0.5 * m, tf.math.square(mag))
    #op = my_print("kinetic energy:", x)
    return x

# @tf.function
def calc_pe(vel, pos):
    x = 0.5*k*pos[0]*pos[0]
    #my_print("potential energy:", x)
    return x

# @tf.function
def calc_total_energy(vel, pos):
    return ke(vel) + pe(pos)

# @tf.function
def update_force(pos, vel, force, delta_t):
    return tf.math.scalar_mul(-k, pos)

#@tf.function
def update_vel(pos, vel, force, delta_t):
    t = tf.math.scalar_mul(0.5 * delta_t / m, force)
    return t + vel

# @tf.function
def update_pos(pos, vel, force, delta_t):
    t = tf.math.scalar_mul(delta_t, vel)
    return t + pos

# @tf.function
def compute_energies(pos, vel, force, energies, delta_t):
    ke = calc_ke(vel, pos)
    pe = calc_pe(vel, pos)
    tot = ke + pe
    return (ke, pe, tot)

# @tf.function
def loop_control(step_target, step, vecs):
    return step < step_target

@tf.function
def loop_execute(vecs):
    pos, vel, force, energies = vecs
    for i in range(log_freq):
        new_vel = update_vel(pos, vel, force, delta_t)
        new_pos = update_pos(pos, new_vel, force, delta_t)
        new_force = update_force(new_pos, new_vel, force, delta_t)
        new_new_vel = update_vel(new_pos, new_vel, new_force, delta_t)
        energies = compute_energies(new_pos, new_new_vel, new_force, energies, delta_t)
    return (new_pos, new_new_vel, new_force, energies)

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
        sess.run(tf.global_variables_initializer())
        # saver = tf.compat.v1.train.Saver([pos,vel,force,ke,pe,tot], max_to_keep=None)
        # checkpoint = tf.train.Checkpoint(pos=pos, vel=vel, force=force, ke=ke, pe=pe, tot=tot)
        # manager = tf.train.CheckpointManager(checkpoint, directory="/home/alfuerst/simulations/chkpt", max_to_keep=None)

        out = (pos, vel, force, energies)
        for seg in range(num_iterations):
            out = loop_execute(out)
            out = sess.run(out)
            print((1+seg)*log_freq, out)
            # print(saver.save(sess=sess, save_path='checkpoints/{}/'.format((1+seg)*log_freq), global_step=(1+seg)*log_freq))
            # print(manager.save(checkpoint_number=seg))