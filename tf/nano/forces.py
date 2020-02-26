import tensorflow as tf
import numpy as np
import math

import common, utility, interface

@tf.function
def _wrap_distances_on_edges(simul_box, distances):
    """
    Wrap distances on x and y axis. Z axis is unchanged
    """
    with tf.name_scope("wrap_distances_on_edges"):
        edges = tf.constant([simul_box.lx, simul_box.ly, 0], name="box_edges", dtype=common.tf_dtype)
        edges_half = tf.constant([simul_box.lx/2, simul_box.ly/2, 0], name="box_edges_half", dtype=common.tf_dtype)
        neg_edges_half = tf.constant([-simul_box.lx/2, -simul_box.ly/2, 0], name="box_edges_half", dtype=common.tf_dtype)
        wrapped_distances = tf.compat.v1.where_v2(distances > edges_half, distances - edges, distances, name="where_edges_half")
        return tf.compat.v1.where_v2(wrapped_distances < neg_edges_half, wrapped_distances + edges, wrapped_distances, name="where_neg_edges_half")

_tf_zero = tf.constant(0.0, name="const_zero", dtype=common.tf_dtype)
_tf_one = tf.constant(1.0, name="const_one", dtype=common.tf_dtype)
_tf_neg_one = tf.constant(-1.0, name="const_neg_one", dtype=common.tf_dtype)

def _zero_nans(tensor):
    """
    Replaces all nans in the given tensor with 0s
    """
    with tf.name_scope("zero_nans"):
        return tf.compat.v1.where_v2(tf.math.is_nan(tensor), _tf_zero, tensor, name="zero_nans_where")

@tf.function
def _particle_electrostatic_force(simul_box, ion_dict):
    """
    force on the particles (electrostatic)
    parallel calculation of forces (uniform case)
    """
    with tf.name_scope("left_wall_lj_force"):
        distances = tf.compat.v1.vectorized_map(fn=lambda atom_pos: ion_dict["ion_pos"] - atom_pos, elems=ion_dict["ion_pos"])
        z_distances = distances[:, :, -1] # get z-axis value
        abs_z_distances = tf.math.abs(z_distances)
        r1 = tf.math.sqrt(0.5 + (z_distances / simul_box.lx) * (z_distances / simul_box.lx))
        r2 = tf.math.sqrt(0.25 + (z_distances / simul_box.lx) * (z_distances / simul_box.lx))
        E_z = 4 * tf.math.atan(4 * abs_z_distances * r1 / simul_box.lx)

        factor = tf.compat.v1.where_v2(z_distances >= 0.0, _tf_one, _tf_neg_one, name="where_factor")
        hcsh = (4 / simul_box.lx) * (1 / (r1 * (0.5 + r1)) - 1 / (r2 * r2)) * z_distances + factor * E_z + \
                       16 * abs_z_distances * (simul_box.lx / (simul_box.lx * simul_box.lx + 16 * z_distances * z_distances * r1 * r1)) * \
                       (abs_z_distances * z_distances / (simul_box.lx * simul_box.lx * r1) + factor * r1) # MATHEMATICAL

        #h1.z = h1.z + 2 * ion[i].q * (ion[j].q / (box.lx * box.lx)) * 0.5 * (1 / ion[i].epsilon + 1 / ion[j].epsilon) * hcsh
        one_over_ep = 1 / ion_dict["ion_diconst"]
        q_over_lx_sq = ion_dict["ion_charges"] / (simul_box.lx * simul_box.lx)
        vec_one_over_ep = tf.compat.v1.vectorized_map(fn=lambda disconst_j: one_over_ep + disconst_j, elems=one_over_ep)
        vec_q_over_lx_sq = tf.compat.v1.vectorized_map(fn=lambda charges_j: ion_dict["ion_charges"] * charges_j, elems=q_over_lx_sq)

        h1_z = 2 * vec_q_over_lx_sq * 0.5 * (vec_one_over_ep) * hcsh
        h1_z = tf.math.reduce_sum(h1_z, axis=1, keepdims=True)
        
        # h1 =h1+ ((temp_vec ^ ((-1.0) / r3)) ^ ((-0.5) * ion[i].q * ion[j].q * (1 / ion[i].epsilon + 1 / ion[j].epsilon)));
        wrapped_distances = _wrap_distances_on_edges(simul_box, distances)
        r = common.magnitude(wrapped_distances, keepdims=True) # keep third dimension to divide third dim in wrapped_distances later
        r3 = tf.math.pow(r, 3)

        vec_q_mul = tf.compat.v1.vectorized_map(fn=lambda charges_j: ion_dict["ion_charges"] * charges_j, elems=ion_dict["ion_charges"])
        a = _zero_nans(wrapped_distances * ((-1.0) / r3)) # r3 can have zeroes in it, so remove the nans that come from div by zero
        b = ((-0.5) * vec_q_mul * vec_one_over_ep)
        h1 = tf.math.reduce_sum(a * b[:,:,tf.newaxis], axis=1, keepdims=False, name="sum_a_times_b")
        h1_x_y = h1[:,0:2]
        c = h1[:,2:3] + h1_z
        con = tf.concat(values=[h1_x_y, c], axis=1, name="x_y_and_c_concatenate")
        return con * utility.scalefactor

@tf.function
def _right_wall_lj_force(simul_box, ion_dict):
    """
    interaction with the right plane hard wall
    make a dummy particle with the same diameter as the ion and touching right of the right wall s. t. it is closest to the ion
    """
    with tf.name_scope("right_wall_lj_force"):
        # TODO/QUESTION: Is this necessary?
        # if (ion[i].posvec.z > 0.5 * box.lz - ion[i].diameter)  // avoiding calculating interactions between right wall and ions in bulk.
        dummy_mult = tf.constant([1, 1, 0.5*simul_box.lz], name="dummy_mult_right", dtype=common.tf_dtype)
        dummy_pos = ion_dict["ion_pos"] * dummy_mult
        distances = ion_dict["ion_pos"] - dummy_pos
        mag_squared = common.magnitude_squared(distances, axis=1, keepdims=True) * 0.5 # keep 1th dimension to match up with distances later
        diam_2 = tf.math.pow(ion_dict["ion_diameters"] * 0.5, 2.0, name="diam_2_pow")[:, tf.newaxis] # add new dimension to match up with distances later
        d_six = tf.math.pow(diam_2, 3.0, name="diam_6_pow") / tf.math.pow(mag_squared, 3.0, name="mag_6_pow") # magnitude is alread "squared" so only need N/2 power
        d_twelve = tf.math.pow(diam_2, 6.0, name="diam_12_pow") / tf.math.pow(mag_squared, 6.0, name="mag_12_pow")
        slice_forces = distances * (48.0 * utility.elj * (((d_twelve - 0.5 * d_six) * (1.0/mag_squared))))
        return tf.compat.v1.where_v2(mag_squared < (diam_2*utility.dcut2), _tf_zero, slice_forces, name="where_d_cut")

@tf.function
def _particle_lj_force(simul_box, ion_dict):
    """
    excluded volume interactions given by purely repulsive LJ
    ion-ion
    """
    with tf.name_scope("particle_lj_force"):
        distances = tf.compat.v1.vectorized_map(fn=lambda atom_pos: ion_dict["ion_pos"] - atom_pos, elems=ion_dict["ion_pos"])
        diams_sum = tf.compat.v1.vectorized_map(fn=lambda atom_pos: ion_dict["ion_diameters"] + atom_pos, elems=ion_dict["ion_diameters"]) * 0.5
        diams_sum = diams_sum[:,:,tf.newaxis] # add third dimension to match with wrapped_distances and mag_squared later
        wrapped_distances = _wrap_distances_on_edges(simul_box, distances)
        mag_squared = common.magnitude_squared(wrapped_distances, keepdims=True) # keep third dimension to match with wrapped_distances
        #TODO/QUESTION: this calculation is different than original LJ, ask why
        diam_2 = tf.math.pow(diams_sum, 2.0, name="square_diam_diff")
        d_six = tf.math.pow(diams_sum, 6.0, name="diam_6_pow") / tf.math.pow(mag_squared, 3.0, name="mag_6_pow") # magnitude is alread "squared" so only need N/2 power
        d_twelve = tf.math.pow(diams_sum, 12.0, name="diam_12_pow") / tf.math.pow(mag_squared, 6.0, name="mag_12_pow")
        slice_forces = wrapped_distances * (48.0 * utility.elj * (((d_twelve - 0.5 * d_six) * (1.0/mag_squared))))
        # handle case distances pos - atom_pos == 0, causing inf and nan to appear in that position
        filter = tf.math.logical_or(tf.math.is_nan(slice_forces), mag_squared < (diam_2*utility.dcut2), name="or")
        filtered = tf.compat.v1.where_v2(filter, _tf_zero, slice_forces, name="where_or")
        return tf.math.reduce_sum(filtered, axis=0)
    
@tf.function
def _left_wall_lj_force(simul_box, ion_dict):
    """
    ion-box
    interaction with the left plane hard wall
    make a dummy particle with the same diameter as the ion and touching left of the left wall s. t. it is closest to the ion
    """
    with tf.name_scope("left_wall_lj_force"):
        # TODO/QUESTION: Is this necessary?
        # if (ion[i].posvec.z < -0.5 * box.lz + ion[i].diameter)   // avoiding calculating interactions between left wall and ions in bulk.
        dummy_mult = tf.constant([1, 1, -0.5*simul_box.lz], name="dummy_mult_left", dtype=common.tf_dtype)
        dummy_pos = ion_dict["ion_pos"] * dummy_mult
        distances = ion_dict["ion_pos"] - dummy_pos
        mag_squared = common.magnitude_squared(distances, axis=1, keepdims=True) * 0.5 # keep 1th dimension to match up with distances later
        diam_2 = tf.math.pow(ion_dict["ion_diameters"] * 0.5, 2.0, name="diam_2_pow")[:, tf.newaxis] # add new dimension to match up with distances later
        d_six = tf.math.pow(diam_2, 3.0, name="diam_6_pow") / tf.math.pow(mag_squared, 3.0, name="mag_6_pow") # magnitude is alread "squared" so only need N/2 power
        d_twelve = tf.math.pow(diam_2, 6.0, name="diam_12_pow") / tf.math.pow(mag_squared, 6.0, name="mag_12_pow")
        print("d_twelve", d_twelve)
        print("d_six", d_six)
        print("mag_squared", mag_squared)
        slice_forces = distances * (48.0 * utility.elj * (((d_twelve - 0.5 * d_six) * (1.0/mag_squared))))
        return tf.compat.v1.where_v2(mag_squared < (diam_2*utility.dcut2), _tf_zero, slice_forces, name="where_d_cut")

def _electrostatic_wall_force(simul_box, distances, wall_dictionary):
    """
    ion interacting via electrostatic force with discrete planar wall
    """
    with tf.name_scope("electrostatic_wall_force"):
        wall_distances = tf.compat.v1.vectorized_map(fn=lambda atom_pos: atom_pos - wall_dictionary["posvec"], elems=ion_dict["ion_pos"])
        wall_z_dist = wall_distances[:, :, -1] # get z-axis value
        factor = tf.compat.v1.where_v2(wall_z_dist >= 0.0, _tf_one, _tf_neg_one, name="where_factor")
        r1_rightwall = tf.math.sqrt(0.5 + (wall_z_dist / simul_box.lx) * (wall_z_dist / simul_box.lx))
        r2_rightwall = tf.math.sqrt(0.25 + (wall_z_dist / simul_box.lx) * (wall_z_dist / simul_box.lx))

        E_z_rightwall = 4 * tf.math.atan(4 * tf.math.abs(wall_z_dist) * r1_rightwall / simul_box.lx)
        hcsh_rightwall = (4 / simul_box.lx) * (1 / (r1_rightwall * (0.5 + r1_rightwall)) - 1 / (r2_rightwall * r2_rightwall)) * wall_z_dist + factor * E_z_rightwall +\
               16 * tf.math.abs(wall_z_dist) * (simul_box.lx / (simul_box.lx * simul_box.lx + 16 * wall_z_dist * wall_z_dist * r1_rightwall * r1_rightwall)) *\
               (tf.math.abs(wall_z_dist) * wall_z_dist / (simul_box.lx * simul_box.lx * r1_rightwall) + factor * r1_rightwall)

        # h1_rightwall.z = h1_rightwall.z + 2 * ion[i].q * (wall_dummy.q / (box.lx * box.lx)) * 0.5 * (1 / ion[i].epsilon + 1 / wall_dummy.epsilon) * hcsh_rightwall;
        ion_one_over_ep = 1 / ion_dict["ion_diconst"] # 1 / ion[i].epsilon
        wall_one_over_ep = 1 / wall_dictionary["epsilon"] # 1 / wall_dummy.epsilon
        q_over_lx_sq = wall_dictionary["q"] / (simul_box.lx * simul_box.lx) # (wall_dummy.q / (box.lx * box.lx))
        vec_one_over_ep = tf.compat.v1.vectorized_map(fn=lambda ion_eps: wall_one_over_ep + ion_eps, elems=ion_one_over_ep) # (1 / ion[i].epsilon + 1 / wall_dummy.epsilon)
        vec_q_over_lx_sq = tf.compat.v1.vectorized_map(fn=lambda charges_j: q_over_lx_sq * charges_j, elems=ion_dict["ion_charges"]) # ion[i].q * (wall_dummy.q / (box.lx * box.lx))
        
        h1_z = 2 * vec_q_over_lx_sq * 0.5 * (vec_one_over_ep) * hcsh_rightwall
        h1_z = tf.math.reduce_sum(h1_z, axis=1, keepdims=True, name="sum_h1_z")

        # h1_rightwall = h1_rightwall+ ((temp_vec_rightwall ^ ((-1.0) / r3_rightwall)) ^ ((-0.5) * ion[i].q * wall_dummy.q * (1 / ion[i].epsilon + 1 / wall_dummy.epsilon)));
        wrapped_distances = _wrap_distances_on_edges(simul_box, wall_distances)
        r = common.magnitude(wrapped_distances, keepdims=True) # keep third dimension to divide third dim in wrapped_distances later
        r3 = tf.math.pow(r, 3.0, name="r_3")

        vec_q_mul = tf.compat.v1.vectorized_map(fn=lambda charges_j: wall_dictionary["q"] * charges_j, elems=ion_dict["ion_charges"])
        a = _zero_nans(wrapped_distances * ((-1.0) / r3)) * ((-0.5) * vec_q_mul * vec_one_over_ep)[:,:,tf.newaxis] 
        h1 = tf.math.reduce_sum(a, axis=1, keepdims=False, name="sum_a_mul_b")
        z = h1[:,2:3] + h1_z
        con = tf.concat(values=[h1[:,0:2], z], axis=1, name="h1x_y_and_h1_z_concatenate")
        return con * utility.scalefactor

@tf.function
def _electrostatic_right_wall_force(simul_box, ion_dict):
    """
    ion interacting with discretized right wall
    electrostatic between ion and rightwall
    """
    with tf.name_scope("electrostatic_right_wall_force"):
        return _electrostatic_wall_force(simul_box, ion_dict, simul_box.tf_right_plane)

@tf.function
def _electrostatic_left_wall_force(simul_box, ion_dict):
    """
    ion interacting with discretized left wall
    electrostatic between ion and left wall
    """
    with tf.name_scope("electrostatic_left_wall_force"):
        return _electrostatic_wall_force(simul_box, ion_dict, simul_box.tf_left_plane)

@tf.function
def for_md_calculate_force(simul_box, thermostats, ion_dict, bins, charge_meshpoint: float, valency_counterion: int):
    pef = _particle_electrostatic_force(simul_box, tf_ion_real)
    plj = _particle_lj_force(simul_box, tf_ion_real)
    lw_lj = _left_wall_lj_force(simul_box, tf_ion_real)
    rw_lj = _right_wall_lj_force(simul_box, tf_ion_real)
    erw = _electrostatic_right_wall_force(simul_box, tf_ion_real)
    elw = _electrostatic_left_wall_force(simul_box, tf_ion_real)
    return pef + plj + lw_lj + rw_lj + erw + elw
    # TODO/QUESTION: What is 'all_gather' function??

if __name__ == "__main__":
    np.random.seed(0)
    from tensorflow_manip import silence, toggle_cpu
    silence()
    sess = tf.compat.v1.Session()
    sess.as_default()
    utility.unitlength = 1
    utility.scalefactor = utility.epsilon_water * utility.lB_water / utility.unitlength
    bz = 3
    by = bx = math.sqrt(212 / 0.6022 / 0.5 / bz)
    surface_area = bx * by * pow(10.0,-18) # in unit of squared meter
    fraction_diameter = 0.02
    number_meshpoints = pow((1.0/fraction_diameter), 2.0)
    charge_density = -0.001
    valency_counterion = 1
    charge_meshpoint = (charge_density * surface_area) / (utility.unitcharge * number_meshpoints) # in unit of electron charge
    total_surface_charge = charge_meshpoint * number_meshpoints # in unit of electron charge
    counterions =  2.0 * (int(abs(total_surface_charge)/valency_counterion)) # there are two charged surfaces, we multiply the counter ions by two
    print(counterions)
    simul_box = interface.Interface(salt_conc_in=0.5, salt_conc_out=0, salt_valency_in=1, salt_valency_out=1, bx=bx, by=by, bz=bz, initial_ein=1, initial_eout=1)
    smaller_ion_diam = 0.474
    bigger_ion_diam = 0.627
    simul_box.discretize(smaller_ion_diam / utility.unitlength, fraction_diameter, charge_meshpoint)
    ion_dict = simul_box.put_saltions_inside(pz=1, nz=-1, concentration=0.5, positive_diameter_in=smaller_ion_diam, negative_diameter_in=bigger_ion_diam, \
                            counterions=counterions, valency_counterion=1, counterion_diameter_in=smaller_ion_diam, bigger_ion_diameter=bigger_ion_diam)
    tf_ion_real, tf_ion_place = common.make_tf_versions_of_dict(ion_dict)
    sess.run(tf.compat.v1.global_variables_initializer())
    feed = common.create_feed_dict((simul_box.left_plane, simul_box.tf_place_left_plane), (simul_box.right_plane, simul_box.tf_place_right_plane), (ion_dict, tf_ion_place))

    pef = _particle_electrostatic_force(simul_box, tf_ion_real)
    check_op = tf.compat.v1.add_check_numerics_ops()
    print("\npef",pef)
    pef, _ = sess.run(fetches=[pef,check_op], feed_dict=feed)
    print(pef)
    if(pef.shape != tf_ion_real["ion_pos"].shape):
        raise Exception("bad shape {}".format(pef.shape))

    plj = _particle_lj_force(simul_box, tf_ion_real)
    print("\nplj", plj)
    print(plj.eval(session=sess))
    if(plj.shape != tf_ion_real["ion_pos"].shape):
        raise Exception("bad shape {}".format(plj.shape))

    lw_lj = _left_wall_lj_force(simul_box, tf_ion_real)
    print("\nleft", lw_lj)
    lw_lj = lw_lj.eval(session=sess)
    print(lw_lj)
    if (lw_lj[:,0:2] != 0).any():
        raise Exception("Got a non-zero force in x or y direction on left wall forces")
    if(lw_lj.shape != tf_ion_real["ion_pos"].shape):
        raise Exception("bad shape {}".format(lw_lj.shape))

    rw_lj = _right_wall_lj_force(simul_box, tf_ion_real)
    print("\nright", rw_lj)
    rw_lj = rw_lj.eval(session=sess)
    print(rw_lj)
    if (rw_lj[:,0:2] != 0).any():
        raise Exception("Got a non-zero force in x or y direction on right wall forces")
    if(rw_lj.shape != tf_ion_real["ion_pos"].shape):
        raise Exception("bad shape {}".format(rw_lj.shape))

    erw = _electrostatic_right_wall_force(simul_box, tf_ion_real)
    print("\nerw", erw)
    erw = erw.eval(session=sess)
    print(erw)
    if(erw.shape != tf_ion_real["ion_pos"].shape):
        raise Exception("bad shape {}".format(erw.shape))

    elw = _electrostatic_left_wall_force(simul_box, tf_ion_real)
    print("\nelw", elw)
    elw = elw.eval(session=sess)
    print(elw)
    if(elw.shape != tf_ion_real["ion_pos"].shape):
        raise Exception("bad shape {}".format(elw.shape))