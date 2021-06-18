import tensorflow as tf
import numpy as np
import common, utility, interface


def initialize_forces(ion_dict):
    ion_dict[interface.ion_for_str] = np.zeros(ion_dict[interface.ion_pos_str].shape, dtype=common.np_dtype)
    return ion_dict

_tf_zero = tf.constant(0.0, name="const_zero", dtype=common.tf_dtype)
_tf_one = tf.constant(1.0, name="const_one", dtype=common.tf_dtype)
_tf_neg_one = tf.constant(-1.0, name="const_neg_one", dtype=common.tf_dtype)

def _zero_nans(tensor):
    """
    Replaces all nans in the given tensor with 0s
    """
    with tf.name_scope("zero_nans"):
        return tf.compat.v1.where_v2(tf.math.is_nan(tensor), _tf_zero, tensor, name="zero_nans_where")

def for_md_calculate_force(simul_box, ion_dict, charge_meshpoint):
    """
    Updates the forces acting on each ion and returns the updated ion_dict
    """
    with tf.name_scope("calculate_force"):
        pef = _particle_electrostatic_force(simul_box, ion_dict)
        plj = _particle_lj_force(simul_box, ion_dict)
        lw_lj = _left_wall_lj_force(simul_box, ion_dict)
        rw_lj = _right_wall_lj_force(simul_box, ion_dict)
        if abs(charge_meshpoint) != 0:
            erw = _electrostatic_right_wall_force(simul_box, ion_dict)
            elw = _electrostatic_left_wall_force(simul_box, ion_dict)
        else:
            erw = 0.0
            elw = 0.0
        ion_dict[interface.ion_for_str] = plj + lw_lj + rw_lj + erw + elw + pef
        return ion_dict

def _particle_electrostatic_force(simul_box, ion_dict):
    """
    force on the particles (electrostatic)
    parallel calculation of forces (uniform case)
    """
    with tf.name_scope("particle_electrostatic_force"):
        distances = common.wrap_vectorize(fn=lambda atom_pos: atom_pos - ion_dict[interface.ion_pos_str],
                                          elems=ion_dict[interface.ion_pos_str])
        z_distances = distances[:, :,
                      -1]  # get z-axis value #TODO: Remove the need for third axis/pulling out z dimension => see if faster way
        abs_z_distances = tf.math.abs(z_distances)
        r1 = tf.math.sqrt(0.5 + ((z_distances / simul_box.lx) * (z_distances / simul_box.lx)))
        r2 = tf.math.sqrt(0.25 + ((z_distances / simul_box.lx) * (z_distances / simul_box.lx)))
        E_z = 4 * tf.math.atan(4 * abs_z_distances * r1 / simul_box.lx)
        condition = tf.equal(z_distances, 0)
        r1 = tf.compat.v1.where_v2(condition, z_distances, r1, name="r1_cleanup")
        r2 = tf.compat.v1.where_v2(condition, z_distances, r2, name="r2_cleanup")
        factor = tf.compat.v1.where_v2(z_distances >= 0.0, _tf_one, _tf_neg_one, name="where_factor")
        hcsh = (4 / simul_box.lx) * (1 / (r1 * (0.5 + r1)) - 1 / (r2 * r2)) * z_distances + factor * E_z + \
               16 * abs_z_distances * (
                           simul_box.lx / (simul_box.lx * simul_box.lx + 16 * z_distances * z_distances * r1 * r1)) * \
               (abs_z_distances * z_distances / (simul_box.lx * simul_box.lx * r1) + factor * r1)
        hcsh = _zero_nans(hcsh)
        one_over_ep = 1 / ion_dict[interface.ion_epsilon_str]
        q_over_lx_sq = ion_dict[interface.ion_charges_str] / (simul_box.lx * simul_box.lx)
        vec_one_over_ep = common.wrap_vectorize(fn=lambda epsilon_j: epsilon_j + one_over_ep, elems=one_over_ep)
        vec_q_over_lx_sq = common.wrap_vectorize(fn=lambda q_j: q_j * ion_dict[interface.ion_charges_str],
                                                 elems=q_over_lx_sq)
        h1_z = 2 * vec_q_over_lx_sq * 0.5 * vec_one_over_ep * hcsh
        h1_z = tf.math.reduce_sum(h1_z, axis=1, keepdims=True)
        wrapped_distances = common.wrap_distances_on_edges(simul_box, distances)
        r = tf.norm(wrapped_distances, ord='euclidean', axis=2, keepdims=True)
        r3 = tf.math.pow(r, 3)
        vec_q_mul = common.wrap_vectorize(fn=lambda q_j: q_j * ion_dict[interface.ion_charges_str],
                                          elems=ion_dict[interface.ion_charges_str])
        a = _zero_nans(wrapped_distances * (
                    (-1.0) / r3))  # r3 can have zeroes in it, so remove the nans that come from div by zero
        b = ((-0.5) * vec_q_mul * vec_one_over_ep)
        h1 = tf.math.reduce_sum(a * b[:, :, tf.newaxis], axis=1, keepdims=False,
                                name="sum_a_times_b")  # TODO: remove need for newaxis here  #-------------->>>>> change axis here to 0
        h1_x_y = h1[:, 0:2]  # TODO: replace this junk with better impl
        c = h1[:, 2:3] + h1_z
        con = tf.concat(values=[h1_x_y, c], axis=1, name="x_y_and_c_concatenate")
        return con * utility.scalefactor
        # return con * utility.scalefactor, distances, h1, h1_z, hcsh, a, b


def _electrostatic_wall_force(simul_box, ion_dict, wall_dictionary):
    """
    ion interacting via electrostatic force with discrete planar wall
    """
    with tf.name_scope("electrostatic_wall_force"):
        wall_distances = common.wrap_vectorize(fn=lambda atom_pos: atom_pos - wall_dictionary["posvec"],
                                               elems=ion_dict[interface.ion_pos_str])
        wall_z_dist = wall_distances[:, :, -1]  # get z-axis value
        factor = tf.compat.v1.where_v2(wall_z_dist >= 0.0, _tf_one, _tf_neg_one, name="where_factor")
        r1_rightwall = tf.math.sqrt(0.5 + (wall_z_dist / simul_box.lx) * (wall_z_dist / simul_box.lx))
        r2_rightwall = tf.math.sqrt(0.25 + (wall_z_dist / simul_box.lx) * (wall_z_dist / simul_box.lx))
        E_z_rightwall = 4 * tf.math.atan(4 * tf.math.abs(wall_z_dist) * r1_rightwall / simul_box.lx)
        hcsh_rightwall = (4 / simul_box.lx) * (1 / (r1_rightwall * (0.5 + r1_rightwall)) - 1 / (
                    r2_rightwall * r2_rightwall)) * wall_z_dist + factor * E_z_rightwall + 16 * tf.math.abs(
            wall_z_dist) * (simul_box.lx / (
                simul_box.lx * simul_box.lx + 16 * wall_z_dist * wall_z_dist * r1_rightwall * r1_rightwall)) * (
                                     tf.math.abs(wall_z_dist) * wall_z_dist / (
                                         simul_box.lx * simul_box.lx * r1_rightwall) + factor * r1_rightwall)
        ion_one_over_ep = 1 / ion_dict[interface.ion_epsilon_str]  # 1 / ion[i].epsilon
        wall_one_over_ep = 1 / wall_dictionary["epsilon"]  # 1 / wall_dummy.epsilon
        q_over_lx_sq = wall_dictionary["q"] / (simul_box.lx * simul_box.lx)  # (wall_dummy.q / (box.lx * box.lx))
        vec_one_over_ep = common.wrap_vectorize(fn=lambda ion_eps: wall_one_over_ep + ion_eps,
                                                elems=ion_one_over_ep)  # (1 / ion[i].epsilon + 1 / wall_dummy.epsilon)
        vec_q_over_lx_sq = common.wrap_vectorize(fn=lambda q_j: q_j * q_over_lx_sq, elems=ion_dict[
            interface.ion_charges_str])  # ion[i].q * (wall_dummy.q / (box.lx * box.lx))
        h1_z = 2 * vec_q_over_lx_sq * 0.5 * (vec_one_over_ep) * hcsh_rightwall
        h1_z = tf.math.reduce_sum(h1_z, axis=1, keepdims=True, name="sum_h1_z")
        wrapped_distances = common.wrap_distances_on_edges(simul_box, wall_distances)
        r = common.magnitude(wrapped_distances,
                             keepdims=True)  # keep third dimension to divide third dim in wrapped_distances later
        r3 = tf.math.pow(r, 3.0, name="r_3")
        vec_q_mul = common.wrap_vectorize(fn=lambda q_j: wall_dictionary["q"] * q_j,
                                          elems=ion_dict[interface.ion_charges_str])
        a = _zero_nans(wrapped_distances * ((-1.0) / r3)) * ((-0.5) * vec_q_mul * vec_one_over_ep)[:, :, tf.newaxis]
        h1 = tf.math.reduce_sum(a, axis=1, keepdims=False, name="sum_a_mul_b")
        z = h1[:, 2:3] + h1_z
        con = tf.concat(values=[h1[:, 0:2], z], axis=1, name="h1x_y_and_h1_z_concatenate")
        return con * utility.scalefactor


def _particle_lj_force(simul_box, ion_dict):
    """
    excluded volume interactions given by purely repulsive LJ
    ion-ion
    """
    with tf.name_scope("particle_lj_force"):
        distances = common.wrap_vectorize(fn=lambda atom_pos: atom_pos - ion_dict[interface.ion_pos_str],
                                          elems=ion_dict[interface.ion_pos_str])
        wrapped_distances = common.wrap_distances_on_edges(simul_box, distances)
        d = common.wrap_vectorize(fn=lambda atom_diam: ion_dict[interface.ion_diameters_str] + atom_diam,
                                  elems=ion_dict[interface.ion_diameters_str]) * 0.5
        d = d[:, :, tf.newaxis]  # add third dimension to match with wrapped_distances and r2 later
        r2 = common.magnitude_squared(wrapped_distances, axis=2,
                                      keepdims=True)  # keep third dimension to match with wrapped_distances
        condition = tf.equal(wrapped_distances, 0)
        d = tf.compat.v1.where_v2(condition, wrapped_distances, d, name="d_cleanup")
        d_2 = tf.math.pow(d, 2.0, name="square_diam_diff")
        d_6 = tf.math.pow(d_2, 3.0, name="diam_6_pow")
        r_6 = tf.math.pow(r2, 3.0, name="r_6_pow")  # magnitude is alread "squared" so only need N/2 power
        d_12 = tf.math.pow(d_2, 6.0, name="diam_12_pow")
        r_12 = tf.math.pow(r2, 6.0, name="r_12_pow")
        slice_forces = wrapped_distances * (48.0 * utility.elj * ((d_12 / r_12) - 0.5 * (d_6 / r_6)) * (1.0 / r2))
        slice_forces = tf.compat.v1.where_v2(tf.math.is_nan(slice_forces), _tf_zero, slice_forces, name="where_nan")
        slice_forces = tf.compat.v1.where_v2(r2 < (utility.dcut2 * d_2), slice_forces, _tf_zero, name="where_dcut")
        return tf.math.reduce_sum(slice_forces, axis=1)


def _left_wall_lj_force(simul_box, ion_dict):
    """
    ion-box
    interaction with the left plane hard wall
    make a dummy particle with the same diameter as the ion and touching left of the left wall s. t. it is closest to the ion
    """
    with tf.name_scope("left_wall_lj_force"):
        mask = ion_dict[interface.ion_pos_str][:, -1] < ((-0.5 * simul_box.lz) + ion_dict[
            interface.ion_diameters_str])  # TODO: remove this mask if not cause of sim error
        dummy_mult = tf.constant([1, 1, 0], name="dummy_mult_left", dtype=common.tf_dtype)
        dummy_pos = ion_dict[interface.ion_pos_str] * dummy_mult
        # TODO!: replace - 0.5 with 0.5* diameter for correctness
        # dummy_add = tf.constant([0, 0, (-0.5 * simul_box.lz) -0.5], name="dummy_add_left", dtype=common.tf_dtype)
        dummy_add = tf.constant([0, 0, (-0.5 * simul_box.lz)], name="dummy_add_left", dtype=common.tf_dtype)
        dummy_pos = dummy_pos + dummy_add
        distances = ion_dict[interface.ion_pos_str] - dummy_pos
        r2 = common.magnitude_squared(distances, axis=1,
                                      keepdims=True)  # keep 1th dimension to match up with distances later
        diam_2 = tf.math.pow(ion_dict[interface.ion_diameters_str] * 0.5, 2.0, name="diam_2_pow")[:,
                 tf.newaxis]  # add new dimension to match up with distances later
        d_r_6 = tf.math.pow(diam_2, 3.0, name="diam_6_pow") / tf.math.pow(r2, 3.0,
                                                                          name="r_6_pow")  # magnitude is alread "squared" so only need N/2 power
        d_r_12 = tf.math.pow(diam_2, 6.0, name="diam_12_pow") / tf.math.pow(r2, 6.0, name="r_12_pow")
        slice_forces = distances * (48.0 * utility.elj * (d_r_12 - 0.5 * d_r_6) * (1.0 / r2))
        d_cut = tf.compat.v1.where_v2(r2 < (diam_2 * utility.dcut2), slice_forces, _tf_zero, name="where_d_cut")
        return tf.compat.v1.where_v2(mask[:, tf.newaxis], d_cut, _tf_zero, name="lj_wall_bulk_cutoff")


def _right_wall_lj_force(simul_box, ion_dict):
    """
    interaction with the right plane hard wall
    make a dummy particle with the same diameter as the ion and touching right of the right wall s. t. it is closest to the ion
    """
    with tf.name_scope("right_wall_lj_force"):
        mask = ion_dict[interface.ion_pos_str][:, -1] > ((0.5 * simul_box.lz) - ion_dict[
            interface.ion_diameters_str])  # TODO: remove this mask if not cause of sim error
        dummy_mult = tf.constant([1, 1, 0], name="dummy_mult_right", dtype=common.tf_dtype)
        dummy_pos = ion_dict[interface.ion_pos_str] * dummy_mult
        # TODO!: replace + 0.5 with 0.5* diameter for correctness
        dummy_add = tf.constant([0, 0, (0.5 * simul_box.lz)], name="dummy_add_right", dtype=common.tf_dtype)
        dummy_pos = dummy_pos + dummy_add
        distances = ion_dict[interface.ion_pos_str] - dummy_pos
        r2 = common.magnitude_squared(distances, axis=1,
                                      keepdims=True)  # keep 1th dimension to match up with distances later
        d2 = tf.math.pow((ion_dict[interface.ion_diameters_str] * 0.5), 2.0, name="d_2_pow")[:,
             tf.newaxis]  # add new dimension to match up with distances later
        d_r_6 = tf.math.pow(d2, 3.0, name="diam_6_pow") / tf.math.pow(r2, 3.0,
                                                                      name="mag_6_pow")  # magnitude is alread "squared" so only need N/2 power
        d_r_12 = tf.math.pow(d2, 6.0, name="diam_12_pow") / tf.math.pow(r2, 6.0, name="r_12_pow")
        slice_forces = distances * (48.0 * utility.elj * (d_r_12 - 0.5 * d_r_6) * (1.0 / r2))
        d_cut = tf.compat.v1.where_v2(r2 < (d2 * utility.dcut2), slice_forces, _tf_zero, name="where_d_cut")
        return tf.compat.v1.where_v2(mask[:, tf.newaxis], d_cut, _tf_zero,
                                     name="lj_wall_bulk_cutoff")  # , distances, dummy_pos   -----revisit this


def _electrostatic_right_wall_force(simul_box, ion_dict):
    """
    ion interacting with discretized right wall
    electrostatic between ion and rightwall
    """
    with tf.name_scope("electrostatic_right_wall_force"):
        return _electrostatic_wall_force(simul_box, ion_dict, simul_box.tf_right_plane)


def _electrostatic_left_wall_force(simul_box, ion_dict):
    """
    ion interacting with discretized left wall
    electrostatic between ion and left wall
    """
    with tf.name_scope("electrostatic_left_wall_force"):
        return _electrostatic_wall_force(simul_box, ion_dict, simul_box.tf_left_plane)



mean_pos_density = mean_sq_pos_density = mean_neg_density = mean_sq_neg_density=[]

