import numpy as np
import tensorflow as tf
import sys
m = 1
k = 1
a = -1
totaltime = 10
steps = 10000
delta_t = totaltime / steps
prints = []
write_op = None
log = "file://home/alfuerst/simulations/log.txt"
tf_log = tf.constant(log)
log_dir = "/home/alfuerst/simulations/"

def my_print(*args):
    op = tf.print(args, output_stream=log)
    prints.append(op)
    return op

def calc_ke(vel, pos):
    squared = tf.math.square(vel)
    sq_sum = tf.math.reduce_sum(squared, tf.constant([0]))
    mag = tf.math.sqrt(sq_sum)
    x = tf.math.scalar_mul(0.5 * m, tf.math.square(mag))
    op = my_print("kinetic energy:", x)
    return x

def calc_pe(vel, pos):
    x = 0.5*k*pos[0]*pos[0]
    my_print("potential energy:", x)
    return x

def calc_total_energy(vel, pos):
    return ke(vel) + pe(pos)

def update_force(pos, vel, force, delta_t):
    return tf.math.scalar_mul(-k, force)

def update_vel(pos, vel, force, delta_t):
    t = tf.math.scalar_mul(0.5 * delta_t * m, force)
    return t + vel

def update_pos(pos, vel, force, delta_t):
    t = tf.math.scalar_mul(delta_t, vel)
    return t + pos

def compute_energies(pos, vel, force, energies, delta_t):
    ke = calc_ke(vel, pos)
    pe = calc_pe(vel, pos)
    tot = ke + pe
    my_print("kinetic energy returned:", k)
    return (ke, pe, tot)

def loop_control(step, vecs):
    num_steps = 10000
    return step < num_steps

def loop_execute(step, vecs):
    pos, vel, force, energies = vecs
    step += 1
    vel = update_vel(pos, vel, force, delta_t)
    pos = update_pos(pos, vel, force, delta_t)
    force = update_force(pos, vel, force, delta_t)
    vel = update_vel(pos, vel, force, delta_t)
    energies = compute_energies(pos, vel, force, energies, delta_t)
    po = my_print(vel)
    global write_op
    write_op = tf.io.write_file(tf_log, tf.strings.format("{}", step))
    print(write_op)
    return step, (pos, vel, force, energies,)

if __name__ == '__main__':
    pos = tf.Variable(np.array([a, 0, 0], dtype=np.float64), name="position")
    vel = tf.Variable(np.array([0, 0, 0], dtype=np.float64), name="velocity")
    force = tf.Variable(np.array([-k*a, 0, 0], dtype=np.float64), name="force")
    ke = tf.Variable(0, dtype=np.float64, name="kinetic")
    pe = tf.Variable(0.5*k*a*a, dtype=np.float64, name="potential")
    tot = tf.Variable(0.5*k*a, dtype=np.float64, name="total")
    energies = (ke, pe, tot)
    step = tf.Variable(0, name="step_counter")
    init_op = tf.compat.v1.global_variables_initializer()

    with tf.compat.v1.Session() as sess:
        sess.as_default()
        #writer = tf.compat.v1.summary.FileWriter(log_dir, sess.graph)
        i, out = tf.while_loop(loop_control, loop_execute, (step, (pos, vel, force, energies)))

        with tf.control_dependencies([write_op]):
            sess.run(init_op)
            sess.run(write_op)
            print(sess.run(i))
            print(sess.run(out))
