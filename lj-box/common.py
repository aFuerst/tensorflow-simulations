import tensorflow as tf

"""
Calculate the magnituge of a Tensor, should be in shape [x,y,z] or [[x,y,z]]
"""
def magnitude(tensor):
    with tf.name_scope("magnitude"):
        return tf.math.sqrt(tf.math.reduce_sum(tf.math.pow(tensor,2.0), axis=1, keepdims=True))