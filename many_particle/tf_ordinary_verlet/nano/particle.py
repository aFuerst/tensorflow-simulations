import interface
import tensorflow as tf
import velocities

import common


def update_position(simul_box, ion_dict, dt: float):
    with tf.name_scope("update_position"):
        ion_dict[interface.ion_pos_str] = ion_dict[interface.ion_pos_str] + ion_dict[velocities.ion_vel_str] * (0.5 * dt)
        ion_dict[interface.ion_pos_str] = common.wrap_distances_on_edges(simul_box, ion_dict[interface.ion_pos_str])
        return ion_dict