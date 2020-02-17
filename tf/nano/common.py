import tensorflow as tf
import numpy as np
"""
Calculate the magnituge of a Tensor, should be in shape [x,y,z] or [[x,y,z]]
"""
def magnitude(tensor):
    with tf.name_scope("magnitude"):
        return tf.math.sqrt(tf.math.reduce_sum(tf.math.pow(tensor,2.0), axis=1, keepdims=True))
        

"""
Calculate the magnituge of a numpy array, should be in shape [x,y,z] or [[x,y,z]]
"""
def magnitude_np(array):
    return np.sqrt(np.sum(np.power(array,2.0)))
        