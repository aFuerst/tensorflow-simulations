import numpy as np
import tensorflow as tf
import sys
m = 1
k = 1
a = -1
totaltime = 10
steps = 10000
log_freq = 100
num_iterations = steps // log_freq
delta_t = totaltime / steps
prints = []
log = "file://home/alfuerst/simulations/log.txt"
tf_log = tf.constant(log)
log_dir = "/home/alfuerst/simulations/"
#from tensorflow.python.framework.ops import disable_eager_execution
#disable_eager_execution()

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
    return tf.math.scalar_mul(-k, force)

#@tf.function
def update_vel(pos, vel, force, delta_t):
    t = tf.math.scalar_mul(0.5 * delta_t * m, force)
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
    #my_print("kinetic energy returned:", k)
    return (ke, pe, tot)

# @tf.function
def loop_control(step, vecs):
    num_steps = log_freq
    return step < num_steps

# @tf.function
def loop_execute(step, vecs):
    pos, vel, force, energies = vecs
    step = step + 1
    vel = update_vel(pos, vel, force, delta_t)
    pos = update_pos(pos, vel, force, delta_t)
    force = update_force(pos, vel, force, delta_t)
    vel = update_vel(pos, vel, force, delta_t)
    print(vel)
    energies = compute_energies(pos, vel, force, energies, delta_t)
    return step, (pos, vel, force, energies)

if __name__ == '__main__':
    pos = tf.Variable(np.array([a, 0, 0], dtype=np.float64), name="position")
    vel = tf.Variable(np.array([0, 0, 0], dtype=np.float64), name="velocity")
    force = tf.Variable(np.array([-k*a, 0, 0], dtype=np.float64), name="force")
    ke = tf.Variable(0, dtype=np.float64, name="kinetic")
    pe = tf.Variable(0.5*k*a*a, dtype=np.float64, name="potential")
    tot = tf.Variable(0.5*k*a, dtype=np.float64, name="total")
    energies = (ke, pe, tot)
    step = tf.Variable(0, name="step_counter")
    
    with tf.compat.v1.Session() as sess:
        sess.as_default()
        sess.run(tf.global_variables_initializer())
        saver = tf.compat.v1.train.Saver([step,pos,vel,force,ke,pe,tot])
        
        out = (pos, vel, force, energies)
        for seg in range(num_iterations):
            # for _ in range(log_freq):
            #     step, out = loop_execute(step, out)
            step, out = tf.while_loop(loop_control, loop_execute, (step, (pos, vel, force, energies)))
            step = step * 0
            print(sess.run(step))
            print(sess.run(out))
            #saver.save_counter = seg*log_freq
            saver.save(sess=sess, save_path='checkpoints/checkpoint', global_step=seg*log_freq, write_meta_graph=False)
            