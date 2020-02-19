import tensorflow as tf
import numpy as np

import utility, bin, forces
from common import *

def run_md_sim(sumil_box, thermostats, ion_dict, bins, charge_meshpoint: float, valency_counterion: int):
    # ion_dict["saltion_pos"]saltion_in_pos, ion_dict["ion_pos"]ion_pos,\
    #                  ion_dict["ion_charges"]ion_charges, ion_dict["ion_masses"]ion_masses, ion_dict["ion_diameters"]ion_diameter, ion_dict["ion_diconst"]ion_discont
    sess = tf.compat.v1.Session()
    sess.as_default()
    ion_p = make_tf_place_of_nparray(ion_dict["ion_pos"], "ion_pos")
    sess.run(tf.compat.v1.global_variables_initializer())
    
    bin_nums_tf = bin.tf_get_ion_bin_density(sumil_box, ion_p, bins) # build 'graph'
    force_g = forces.for_md_calculate_force(sumil_box, thermostats, ion_dict, bins, charge_meshpoint, valency_counterion)


    feed_dict = {ion_p:ion_dict["ion_pos"]}
    bin_nums = sess.run(bin_nums_tf, feed_dict=feed_dict)
    print(bin_nums)