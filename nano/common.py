import tensorflow as tf
import numpy as np

np_dtype = np.float64
tf_dtype = tf.dtypes.float64

def wrap_distances_on_edges(simul_box, distances):
    """
    Wrap distances on x and y axis. Z axis is unchanged
    """
    with tf.name_scope("wrap_distances_on_edges"):
        edges = tf.constant([simul_box.lx, simul_box.ly, 0], name="box_edges", dtype=tf_dtype) # Z-edge can be zero since we set edge_half to also zero nothing happens
        edges_half = tf.constant([simul_box.lx/2, simul_box.ly/2, 0], name="box_edges_half", dtype=tf_dtype)
        neg_edges_half = tf.constant([-simul_box.lx/2, -simul_box.ly/2, 0], name="neg_box_edges_half", dtype=tf_dtype)
        wrapped_distances = tf.compat.v1.where_v2(distances > edges_half, distances - edges, distances, name="where_edges_half")
        # print("\n Printing wrapped distances::")
        # out_wrap_distances = tf.Print(wrapped_distances, [wrapped_distances[0]])
        return tf.compat.v1.where_v2(wrapped_distances < neg_edges_half, wrapped_distances + edges, wrapped_distances, name="where_neg_edges_half")

def magnitude(tensor, axis:int=2, keepdims:bool=False):
    """
    Calculate the magnituge of a Tensor, should be in shape [x,y,z] or [[x,y,z]]
    The dimension specified by 'axis' must be exactly three (3)
    """
    if(tensor.shape[axis] != 3):
        raise Exception("Given axis value '{}' did not have a depth of 3, but was '{}'".format(axis, tensor.shape[axis]))
    with tf.name_scope("magnitude"):
        return tf.math.sqrt(magnitude_squared(tensor, axis, keepdims))
        
def magnitude_squared(tensor, axis:int=2, keepdims:bool=False):
    """
    Calculate the magnituge of a Tensor, should be in shape [x,y,z] or [[x,y,z]]
    The dimension specified by 'axis' must be exactly three (3)
    """
    if(tensor.shape[axis] != 3):
        raise Exception("Given axis value '{}' did not have a depth of 3, but was '{}'".format(axis, tensor.shape[axis]))
    with tf.name_scope("magnitude_squared"):
        return tf.math.reduce_sum(tf.math.pow(tensor,2.0), axis=axis, keepdims=keepdims)


def my_tf_round(x, decimals = 0):
    multiplier = tf.constant(10**decimals, dtype=x.dtype)
    return tf.round(x * multiplier) / multiplier


def magnitude_np(array, axis:int=2):
    """
    Calculate the magnituge of a numpy array, should be in shape [x,y,z] or [[x,y,z]]
    """
    if(array.shape[axis] != 3):
        raise Exception("Given axis value '{}' did not have a depth of 3, but was '{}'".format(
            axis, array.shape[axis]))
    return np.sqrt(np.sum(np.power(array, 2.0), axis=axis))

def make_tf_place_of_single(np_arr, name=None):
    """
    Return TF placeholder version of given numpy array
    """
    return tf.compat.v1.placeholder(dtype=tf_dtype, shape=(), name=name if name is None else name+"_placeholder")

def make_tf_place_of_nparray(np_arr, name=None):
    """
    Return TF placeholder version of given numpy array
    """
    return tf.compat.v1.placeholder(dtype=tf_dtype, shape=np_arr.shape, name=name if name is None else name+"_placeholder")

def make_tf_vers_of_nparray(np_arr, name=None):
    """
    Return TF variable and placeholder version of given numpy array
    """
    variable = tf.compat.v1.Variable(initial_value=np_arr, name=name, shape=np_arr.shape, dtype=tf_dtype)
    placeholder = make_tf_place_of_nparray(np_arr, name)
    return variable, placeholder

def copy_dict(src):
    ret = {}
    for key, value in src.items():
        if type(value) is dict:
            ret[key] = copy_dict(value)
        else:
            ret[key] = value
    return ret

def make_tf_placeholder_of_dict(dictonary):
    """
    Return two identical dictionaries that are TF variables and TF placeholders of the given dictionaries numpy arrays
    """
    tf_placeholders = {}
    for key in dictonary.keys():
        if(type(dictonary[key]) is float or type(dictonary[key]) is int):
            tf_placeholders[key] = make_tf_place_of_single(dictonary[key], key)
        else:      
            tf_placeholders[key] = make_tf_place_of_nparray(dictonary[key], key)
    return tf_placeholders, copy_dict(tf_placeholders)

def make_tf_versions_of_dict(dictonary):
    """
    Return two identical dictionaries that are TF variables and TF placeholders of the given dictionaries numpy arrays
    """
    tf_variables = {}
    tf_placeholders, _ = make_tf_placeholder_of_dict(dictonary)
    for key in dictonary.keys():
        tf_variables[key] = tf.compat.v1.Variable(initial_value=dictonary[key], name=key, shape=dictonary[key].shape, dtype=tf_dtype)

    return tf_variables, tf_placeholders

def py_array_to_np(array, dtype=np_dtype):
    """
    Create a numpy array from the Python one, cast to the given dtyoe
    """
    return np.array(array, dtype=dtype)

def create_feed_dict(*list_of_dict_pairs):
    """
    create tensorflow feed dictionary from a series of (real_data, placeholder_tensor) dictionary pairs
    dictionary pairs must have matching keys associating real & palceholder data
    """
    ret = {}
    for real, placeholder in list_of_dict_pairs:
        for key in real.keys():
            ret[placeholder[key]] = real[key]
    return ret

def throw_if_bad_boundaries(positions, simul_box):
    edge = simul_box.lz/2
    if (positions[:, 2] > edge).any():
        raise Exception("BAD RIGHT", edge, positions[positions[:,2] > edge], np.nonzero(positions[:,2] > edge))
    if (positions[:, 2] < -edge).any():
        raise Exception("BAD LEFT", -edge, positions[positions[:,2] < -edge], np.nonzero(positions[:,2] < -edge))

def wrap_vectorize(fn, elems):
    return tf.function(lambda: tf.compat.v1.vectorized_map(fn=fn, elems=elems))()

