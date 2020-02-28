import tensorflow as tf
import numpy as np

import common, utility, velocities, interface

def _unknown():
    """
    charged sheets method is used to compute Coulomb interactions; an Ewald version should be designed to compare and ensure that long-range effects are taken into account in either methods
    """
    pass

def _lj_energy():
    """
    Excluded volume interaction energy given by purely repulsive LJ
    ion-ion
    """
    pass

def _left_wall_lj_energy():
    """
    left wall
    ion interacting with left wall directly (self, closest)
    """
    pass

def _right_wall_lj_energy():
    """
    right wall
    ion interacting with right wall directly (self, closest)
    """
    pass

def _right_wall_columb_energy():
    """
    coulomb interaction ion-rightwall
    """
    pass

def _left_wall_columb_energy():
    """
    coulomb interaction ion-leftwall
    """
    pass

def kinetic_energy(ion_dict):
    with tf.name_scope("kinetic_energy"):
        ke = 0.5 * ion_dict[interface.ion_masses_str] * common.magnitude_squared(ion_dict[velocities.ion_vel_str], axis=1)
        return tf.reduce_sum(ke)

def np_kinetic_energy(ion_dict):
    ke = 0.5 * ion_dict[interface.ion_masses_str] * np.power(common.magnitude_np(ion_dict[velocities.ion_vel_str]), 2)
    return np.sum(ke)

def energy_functional():
    pass

if __name__ == "__main__":
    from tensorflow_manip import silence, toggle_cpu
    silence()
    sess = tf.compat.v1.Session()
    sess.as_default()

    mass = np.ones((5), dtype=common.np_dtype)
    vel = np.ones((5,3), dtype=common.np_dtype)
    ion_dict = {interface.ion_masses_str:mass, velocities.ion_vel_str:vel}
    tf_ion_real, tf_ion_place = common.make_tf_versions_of_dict(ion_dict)
    sess.run(tf.compat.v1.global_variables_initializer())
    feed = common.create_feed_dict((ion_dict, tf_ion_place))

    ke = kinetic_energy(tf_ion_real)
    check_op = tf.compat.v1.add_check_numerics_ops()
    print("\nke",ke)
    ke, _ = sess.run(fetches=[ke, check_op], feed_dict=feed)
    print(ke)