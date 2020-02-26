import tensorflow as tf
import numpy as np

np_dtype = np.float64
tf_dtype = tf.dtypes.float64

def magnitude(tensor, axis=2, keepdims=False):
    """
    Calculate the magnituge of a Tensor, should be in shape [x,y,z] or [[x,y,z]]
    The dimension specified by 'axis' must be exactly three (3)
    """
    if(tensor.shape[axis] != 3):
        raise Exception("Given axis value '{}' did not have a depth of 3, but was '{}'".format(axis, tensor.shape[axis]))
    with tf.name_scope("magnitude"):
        return tf.math.sqrt(magnitude_squared(tensor, axis, keepdims))
        
def magnitude_squared(tensor, axis=2, keepdims=False):
    """
    Calculate the magnituge of a Tensor, should be in shape [x,y,z] or [[x,y,z]]
    The dimension specified by 'axis' must be exactly three (3)
    """
    if(tensor.shape[axis] != 3):
        raise Exception("Given axis value '{}' did not have a depth of 3, but was '{}'".format(axis, tensor.shape[axis]))
    with tf.name_scope("magnitude_squared"):
        return tf.math.reduce_sum(tf.math.pow(tensor,2.0), axis=axis, keepdims=keepdims)


def magnitude_np(array):
    """
    Calculate the magnituge of a numpy array, should be in shape [x,y,z] or [[x,y,z]]
    """
    return np.sqrt(np.sum(np.power(array,2.0)))

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

def make_tf_placeholder_of_dict(dictonary):
    """
    Return two identical dictionaries that are TF variables and TF placeholders of the given dictionaries numpy arrays
    """
    tf_placeholders = {}
    for key in dictonary.keys():
        tf_placeholders[key] = make_tf_place_of_nparray(dictonary[key], key)
    return tf_placeholders


def make_tf_versions_of_dict(dictonary):
    """
    Return two identical dictionaries that are TF variables and TF placeholders of the given dictionaries numpy arrays
    """
    tf_variables = {}
    tf_placeholders = make_tf_placeholder_of_dict(dictonary)
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

if __name__ == "__main__":
    positions = np.ones((5,3))
    masses = np.ones(5)
    thms = np.ones(5)

    v, p = make_tf_vers_of_nparray(positions)
    print(v, p)
    v, p = make_tf_vers_of_nparray(masses, name="masses")
    print(v, p)
    print()
    test_dict = {"positions":positions, "masses":masses, "thms":thms}
    vs, ps = make_tf_versions_of_dict(test_dict)
    print(vs)
    print(ps)

    print()
    print(py_array_to_np([1.1,2.2,3.3]))
    print(py_array_to_np([1.1,2.2,3.3], dtype=np.int8))

    print("\n\n")
    np.random.seed(0)
    from tensorflow_manip import silence, toggle_cpu
    silence()
    sess = tf.compat.v1.Session()
    sess.as_default()
    distances = np.ones((5,5,3))
    v,p = make_tf_vers_of_nparray(distances)
    mag = magnitude(v)
    mag_sq = magnitude_squared(v)
    
    sess.run(tf.compat.v1.global_variables_initializer())
    print(mag)
    print(sess.run(mag))

    print(mag_sq)
    print(sess.run(mag_sq))