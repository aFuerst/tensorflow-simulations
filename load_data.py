import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
import tensorflow as tf

log_dir = "/home/alfuerst/simulations/checkpoints/1000/-1000"

with tf.compat.v1.Session() as sess:
    sess.as_default()
    pos = tf.Variable(tf.ones([3]))
    vel = tf.Variable(tf.zeros([3]), name="velocity")
    force = tf.Variable(tf.zeros([3]), name="force")
    ke = tf.Variable(0, dtype=np.float64, name="kinetic")
    pe = tf.Variable(0, dtype=np.float64, name="potential")
    tot = tf.Variable(0, dtype=np.float64, name="total")
    sess.run(tf.compat.v1.global_variables_initializer())
    #saver = tf.compat.v1.train.Saver(max_to_keep=None, allow_empty=True)
    #a = saver.restore(sess=sess, save_path=log_dir)
    #var_23 = [v for v in tf.compat.v1.global_variables() if v.name == "position"]
    # print(var_23)
    # print(tf.compat.v1.global_variables())
    # print(sess)
    # print(sess.graph)
    # print(a)

    print(tf.train.list_variables(tf.train.latest_checkpoint("/home/alfuerst/simulations/chkpt")))

    checkpoint = tf.train.Checkpoint(pos=pos, vel=vel, force=force, ke=ke, pe=pe, tot=tot)
    things = checkpoint.restore(tf.train.latest_checkpoint("/home/alfuerst/simulations/chkpt"))
    print(things.assert_consumed())
    print(pos.eval())