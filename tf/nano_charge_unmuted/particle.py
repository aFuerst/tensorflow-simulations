import tensorflow as tf
import numpy as np

import utility, common, velocities, interface

def update_position(simul_box, ion_dict, dt: float):
    with tf.name_scope("update_position"):
        ion_dict[interface.ion_pos_str] = ion_dict[interface.ion_pos_str] + (ion_dict[velocities.ion_vel_str] * dt) #(0.5 * dt)
        # out_pos = tf.Print(ion_dict[interface.ion_pos_str], [ion_dict[interface.ion_pos_str]])
        # print("\n Positions: update_position :: ion_dict[interface.ion_pos_str] = ", out_pos)
        ion_dict[interface.ion_pos_str] = common.wrap_distances_on_edges(simul_box, ion_dict[interface.ion_pos_str])
        # tf.print(ion_dict[interface.ion_pos_str])
        ion_dict[interface.ion_pos_str] = common.my_tf_round(ion_dict[interface.ion_pos_str], 6)
        return ion_dict