import tensorflow as tf
import numpy as np

np_dtype = np.float64
tf_dtype = tf.dtypes.float64

def magnitude(tensor):
    """
    Calculate the magnituge of a Tensor, should be in shape [x,y,z] or [[x,y,z]]
    """
    with tf.name_scope("magnitude"):
        return tf.math.sqrt(tf.math.reduce_sum(tf.math.pow(tensor,2.0), axis=1, keepdims=True))
        

def magnitude_np(array):
    """
    Calculate the magnituge of a numpy array, should be in shape [x,y,z] or [[x,y,z]]
    """
    return np.sqrt(np.sum(np.power(array,2.0)))

def make_tf_versions_of_dict(dictonary):
    """
    Return two identical dictionaries that are TF variables and TF placeholders of the given dictionaries numpy arrays
    """
    tf_variables = {}
    tf_placeholders = {}
    for key in dictonary.keys():
        tf_variables[key] = tf.compat.v1.Variable(initial_value=dictonary[key], name=key, shape=dictonary[key].shape, dtype=tf_dtype)
        tf_placeholders[key] = tf.compat.v1.placeholder(dtype=tf_dtype, shape=dictonary[key].shape, name=key+"_placeholder")

    return tf_variables, tf_placeholders

def make_tf_vers_of_nparray(np_arr, name=None):
    """
    Return TF variable and placeholder version of given numpy array
    """
    variable = tf.compat.v1.Variable(initial_value=np_arr, name=name, shape=np_arr.shape, dtype=tf_dtype)
    placeholder = tf.compat.v1.placeholder(dtype=tf_dtype, shape=np_arr.shape, name=name if name is None else name+"_placeholder")
    return variable, placeholder

def py_array_to_np(array, dtype=np_dtype):
    """
    Create a numpy array from the Python one, cast to the given dtyoe
    """
    return np.array(array, dtype=dtype)

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