import tensorflow as tf
import numpy as np
import math

import common, utility, interface

def _left_wall_lj_force(sumil_box, ion_dict):
    """
    ion-box
    interaction with the left plane hard wall
    make a dummy particle with the same diameter as the ion and touching left of the left wall s. t. it is closest to the ion
    """
    with tf.name_scope("left_wall_lj_force"):
        ion_positions = ion_dict["ion_pos"]
        # ion_positions = ion_positions[0:5]
        # print(ion_positions[:, 2].shape)
        # global sess
        distances = tf.compat.v1.vectorized_map(fn=lambda atom_pos: ion_positions - atom_pos, elems=ion_positions)
        # print(distances[:, :, -1].eval(session=sess))
        z_distances = distances[:, :, -1] # get z-axis value
        abs_z_distances = tf.math.abs(z_distances)
        # print(distances)
        # print(z_distances)
        # exit()
        r1 = tf.math.sqrt(0.5 + (z_distances / sumil_box.lx) * (z_distances / sumil_box.lx))
        r2 = tf.math.sqrt(0.25 + (z_distances / sumil_box.lx) * (z_distances / sumil_box.lx))
        E_z = 4 * tf.math.atan(4 * abs_z_distances * r1 / sumil_box.lx)

        ones = tf.ones(shape=z_distances.shape, dtype=common.tf_dtype, name="ones")
        neg_ones = tf.constant(value=-1, shape=z_distances.shape, dtype=common.tf_dtype, name="negative_ones")
        factor = tf.compat.v1.where_v2(z_distances >= 0, ones, neg_ones, name="where_factor")

        hcsh = (4 / sumil_box.lx) * (1 / (r1 * (0.5 + r1)) - 1 / (r2 * r2)) * z_distances + factor * E_z + \
                       16 * abs_z_distances * (sumil_box.lx / (sumil_box.lx * sumil_box.lx + 16 * z_distances * z_distances * r1 * r1)) * \
                       (abs_z_distances * z_distances / (sumil_box.lx * sumil_box.lx * r1) + factor * r1)

        #TODO: uuhhh... this
        #h1.z = h1.z + 2 * ion[i].q * (ion[j].q / (box.lx * box.lx)) * 0.5 * (1 / ion[i].epsilon + 1 / ion[j].epsilon) * hcsh
        one_over_ep = 1 / ion_dict["ion_diconst"]
        q = ion_dict["ion_charges"]
        q_over_lx_sq = q / (sumil_box.lx * sumil_box.lx)
        vec_one_over_ep = tf.compat.v1.vectorized_map(fn=lambda disconst_j: one_over_ep + disconst_j, elems=one_over_ep)
        vec_q_over_lx_sq = tf.compat.v1.vectorized_map(fn=lambda charges_j: q_over_lx_sq * charges_j, elems=q_over_lx_sq)
        
        # print(vec_q_over_lx_sq)
        # print(vec_one_over_ep)
        # print(hcsh)
        h1_z = + 2 * vec_q_over_lx_sq * 0.5 * (vec_one_over_ep) * hcsh
        
        # print(h1_z)

        edges = tf.constant([sumil_box.lx, sumil_box.ly, 0], name="box_edges", dtype=common.tf_dtype)
        edges_half = tf.constant([sumil_box.lx/2, sumil_box.ly/2, 0], name="box_edges_half", dtype=common.tf_dtype)
        neg_edges_half = tf.constant([-sumil_box.lx/2, -sumil_box.ly/2, 0], name="box_edges_half", dtype=common.tf_dtype)
        wrapped_distances = tf.compat.v1.where_v2(distances > edges_half, distances - edges, distances, name="where_edges_half")
        wrapped_distances = tf.compat.v1.where_v2(wrapped_distances < neg_edges_half, wrapped_distances + edges, wrapped_distances, name="where_neg_edges_half")
        r = common.magnitude(wrapped_distances)
        r3 = tf.math.pow(r, 3)

        #TODO: uuhhh... this
        print(distances)
        print(r)
        # h1 =h1+ ((temp_vec ^ ((-1.0) / r3)) ^ ((-0.5) * ion[i].q * ion[j].q * (1 / ion[i].epsilon + 1 / ion[j].epsilon)));
        vec_q_mul = tf.compat.v1.vectorized_map(fn=lambda charges_j: q * charges_j, elems=q)
        print(vec_q_mul)
        print(vec_one_over_ep)
        h1 = (distances * ((-1.0) / r3)) * ((-0.5) * vec_q_mul * vec_one_over_ep)
        print(h1)

def _right_wall_lj_force():
    """
    interaction with the right plane hard wall
    make a dummy particle with the same diameter as the ion and touching right of the right wall s. t. it is closest to the ion
    """
    pass

def _particle_lj_force():
    """
    excluded volume interactions given by purely repulsive LJ
    ion-ion
    """
    pass

def _particle_electrostatic_force():
    """
    force on the particles (electrostatic)
    parallel calculation of forces (uniform case)
    """
    pass

def _electrostatic_right_wall_force():
    """
    ion interacting with discretized right wall
    electrostatic between ion and rightwall
    """
    pass

def _electrostatic_left_wall_force():
    """
    ion interacting with discretized left wall
    electrostatic between ion and left wall
    """
    pass

def for_md_calculate_force(sumil_box, thermostats, ion_dict, bins, charge_meshpoint: float, valency_counterion: int):
    pass
    
if __name__ == "__main__":
    from tensorflow_manip import silence
    silence()
    global sess
    sess = tf.compat.v1.Session()
    sess.as_default()
    utility.unitlength = 1
    utility.scalefactor = utility.epsilon_water * utility.lB_water / utility.unitlength
    bz = 3
    by = bx = math.sqrt(212 / 0.6022 / 0.5 / bz)
    surface_area = bx * by * pow(10.0,-18) # in unit of squared meter
    fraction_diameter = 0.02
    number_meshpoints = pow((1.0/fraction_diameter), 2.0)
    charge_density = 0.0
    valency_counterion = 1
    charge_meshpoint = (charge_density * surface_area) / (utility.unitcharge * number_meshpoints) # in unit of electron charge
    total_surface_charge = charge_meshpoint * number_meshpoints # in unit of electron charge
    counterions =  2.0 * (int(abs(total_surface_charge)/valency_counterion)) # there are two charged surfaces, we multiply the counter ions by two
    print(counterions)
    sumil_box = interface.Interface(salt_conc_in=0.5, salt_conc_out=0, salt_valency_in=1, salt_valency_out=1, bx=bx, by=by, bz=bz, initial_ein=1, initial_eout=1)
    ion_dict = sumil_box.put_saltions_inside(pz=1, nz=-1, concentration=0.5, positive_diameter_in=0.474, negative_diameter_in=0.627 \
        , counterions=counterions, valency_counterion=1, counterion_diameter_in=0.474, bigger_ion_diameter=0.627)

    tf_ion_real, tf_ion_place = common.make_tf_versions_of_dict(ion_dict)
    sess.run(tf.compat.v1.global_variables_initializer())
    _left_wall_lj_force(sumil_box, tf_ion_real)