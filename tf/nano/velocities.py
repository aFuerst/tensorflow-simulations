import tensorflow as tf
import numpy as np
import math

import utility, common, interface

ion_vel_str = "ion_velocities"

def initialize_particle_velocities(ion_dict, thermostats):
    """
    Numpy implementation to generate random particle starting velocities
    Velocities will all be 0 if there is only one (1) thermostat
    """
    if len(thermostats) == 1: # start with no velocities
        ion_dict[ion_vel_str] =  np.zeros(ion_dict[interface.ion_pos_str].shape, dtype=common.np_dtype) 
    else:
        p_sigma = math.sqrt(utility.kB * utility.T / (2.0 * ion_dict[interface.ion_masses_str][0])) # Maxwell distribution width

        random_vels = np.random.normal(0, p_sigma, ion_dict[interface.ion_pos_str].shape)
        avg_vel = np.average(random_vels, axis=0)
        avg_vel = avg_vel * (1/len(ion_dict[interface.ion_pos_str]))
        ion_dict[ion_vel_str] = random_vels-avg_vel
    return ion_dict

def update_velocity(ion_dict, dt: float, expfac):
    """
    update velocities, expfac should be a TF variable
    """
    with tf.name_scope("update_velocity"):
        ion_dict[ion_vel_str] = (ion_dict[ion_vel_str] * expfac) + (ion_dict[interface.ion_for_str] * (0.5 * dt * tf.math.sqrt(expfac)))
        return ion_dict