import tensorflow as tf
import numpy as np

import common, utility, velocities, interface, thermostat

_tf_zero = tf.constant(0.0, name="const_zero", dtype=common.tf_dtype)


def compute_n_write_useful_data(ion, real_bath, box, charge_meshpoint):
    potential_energy = energy_functional(box, charge_meshpoint, ion)
    particle_ke = np_kinetic_energy(ion)
    # print("KE:", particle_ke)
    real_bath_ke = tf.cast(bath_kinetic_energy(real_bath), tf.float64)
    real_bath_pe = tf.cast(bath_potential_energy(real_bath), tf.float64)
    # extenergy = particle_ke + real_bath_ke + real_bath_pe + potential_energy
    return particle_ke, potential_energy, real_bath_ke, real_bath_pe

def _zero_nans(tensor):
    """
    Replaces all nans in the given tensor with 0s
    """
    with tf.name_scope("zero_nans"):
        return tf.compat.v1.where_v2(tf.math.is_nan(tensor), _tf_zero, tensor, name="zero_nans_where")

def ion_energy(ion_dict, simul_box):
    """
    charged sheets method is used to compute Coulomb interactions; an Ewald version should be designed to compare and ensure that long-range effects are taken into account in either methods
    """
    with tf.name_scope("particle_electrostatic_energy"):
        distances = common.wrap_vectorize(fn=lambda atom_pos: atom_pos - ion_dict[interface.ion_pos_str],elems=ion_dict[interface.ion_pos_str])
        z_distances = distances[:, :,-1]  # get z-axis value #TODO: Remove the need for third axis/pulling out z dimension => see if faster way
        abs_z_distances = tf.math.abs(z_distances)
        r1 = tf.math.sqrt(0.5 + ((z_distances / simul_box.lx) * (z_distances / simul_box.lx)))
        r2 = tf.math.sqrt(0.25 + ((z_distances / simul_box.lx) * (z_distances / simul_box.lx)))
        condition = tf.equal(z_distances, 0)
        r1 = tf.compat.v1.where_v2(condition, z_distances, r1, name="r1_cleanup")
        r2 = tf.compat.v1.where_v2(condition, z_distances, r2, name="r2_cleanup")
        E_z = 4 * tf.math.atan(4 * abs_z_distances * r1 / simul_box.lx)
        fcsh_z = 4 * simul_box.lx * tf.math.log((0.5+r1)/r2) - abs_z_distances * (2* utility.pi - E_z)
        fcsh_inf = -2 * utility.pi * abs_z_distances
        # fqq_csh_ion = ion_dict[interface.ion_charges_str] * ()
        # factor = tf.compat.v1.where_v2(z_distances >= 0.0, _tf_one, _tf_neg_one, name="where_factor")

        # THIS HCSH MIGHT BE INCORRECT HENCE (3,3,3) IS CAUSING A TROUBLE
        # hcsh = (4 / simul_box.lx) * (1 / (r1 * (0.5 + r1)) - 1 / (r2 * r2)) * z_distances + factor * E_z + 16 * abs_z_distances * (
        #                    simul_box.lx / (simul_box.lx * simul_box.lx + 16 * z_distances * z_distances * r1 * r1)) * (abs_z_distances * z_distances / (simul_box.lx * simul_box.lx * r1) + factor * r1)
        # hcsh = _zero_nans(hcsh)
        one_over_ep = 1 / ion_dict[interface.ion_epsilon_str]
        q_over_lx_sq = ion_dict[interface.ion_charges_str] / (simul_box.lx * simul_box.lx)
        vec_one_over_ep = common.wrap_vectorize(fn=lambda epsilon_j: epsilon_j + one_over_ep, elems=one_over_ep)

        vec_q_over_lx_sq = common.wrap_vectorize(fn=lambda q_j: ion_dict[interface.ion_charges_str] * q_j,elems=q_over_lx_sq)
        fqq_csh = vec_q_over_lx_sq * 0.5 * vec_one_over_ep * (fcsh_inf - fcsh_z)
        fqq_csh_sum = tf.math.reduce_sum(fqq_csh, axis=1, keepdims=True)

        wrapped_distances = common.wrap_distances_on_edges(simul_box, distances)
        r = tf.norm(wrapped_distances, ord='euclidean', axis=2, keepdims=True)
        vec_q_mul = common.wrap_vectorize(fn=lambda q_j: q_j * ion_dict[interface.ion_charges_str], elems=ion_dict[interface.ion_charges_str])
        # a = _zero_nans(wrapped_distances * ((-1.0) / r3))  # r3 can have zeroes in it, so remove the nans that come from div by zero
        b = (0.5 * vec_q_mul * 0.5 * vec_one_over_ep)
        fqq = _zero_nans(b[:, :, tf.newaxis]/r)
        fqq = tf.compat.v1.where_v2(tf.math.is_inf(fqq, name="ccheck_in_values"), _tf_zero, fqq, name="filter_infs")
        fqq_sum = tf.math.reduce_sum(fqq, axis=1, keepdims=False)
        # h1 = tf.math.reduce_sum(a * b[:, :, tf.newaxis], axis=1, keepdims=False,
        #                         name="sum_a_times_b")  # TODO: remove need for newaxis here  #-------------->>>>> change axis here to 0
        fqq_sum_xy = fqq_sum[:, 0:2]  # TODO: replace this junk with better impl
        c = fqq_sum[:, 2:3] + fqq_csh_sum
        con = tf.concat(values=[fqq_sum_xy, c], axis=1, name="x_y_and_c_concatenate")
        return common.my_tf_round(tf.math.reduce_sum((con) * utility.scalefactor), 6)


def _lj_energy(ion_dict, simul_box):
    """
    Excluded volume interaction energy given by purely repulsive LJ
    ion-ion
    """
    with tf.name_scope("particle_lj_energy"):
        distances = common.wrap_vectorize(fn=lambda atom_pos: atom_pos - ion_dict[interface.ion_pos_str],
                                          elems=ion_dict[interface.ion_pos_str])
        d = common.wrap_vectorize(fn=lambda atom_diam: ion_dict[interface.ion_diameters_str] + atom_diam,
                                  elems=ion_dict[interface.ion_diameters_str]) * 0.5
        d = d[:, :, tf.newaxis]  # add third dimension to match with wrapped_distances and r2 later
        wrapped_distances = common.wrap_distances_on_edges(simul_box, distances)
        condition = tf.equal(distances, 0)
        wrapped_distances = tf.compat.v1.where_v2(condition, distances, wrapped_distances,
                                                  name="wrapped_distances_cleanup")
        r = common.magnitude(wrapped_distances, axis=2,
                                      keepdims=True)  # keep third dimension to match with wrapped_distances
        condition = tf.equal(r, 0)
        d = tf.compat.v1.where_v2(condition, r, d, name="d_cleanup")

        d_2 = tf.math.pow(d, 2.0, name="square_diam_diff")
        d_6 = tf.math.pow(d_2, 3.0, name="diam_6_pow")
        r_6 = tf.math.pow(r, 6.0, name="r_6_pow")  # magnitude is alread "squared" so only need N/2 power
        # d_12 = tf.math.pow(d_2, 6.0, name="diam_12_pow")
        # r_12 = tf.math.pow(r2, 6.0, name="r_12_pow")

        uljcc = 4 * utility.elj * (d_6 / r_6) * ((d_6 / r_6) - 1) + utility.elj
        # slice_forces = wrapped_distances * (48.0 * utility.elj * ((d_12 / r_12) - 0.5 * (d_6 / r_6)) * (1.0 / r2))

        # handle case distances pos - atom_pos == 0, causing inf and nan to appear in that position
        # slice_forces = tf.compat.v1.debugging.check_numerics(slice_forces, message="slice_forces lj forces")
        # filter = tf.math.logical_or(tf.math.is_nan(slice_forces), r2 >= (utility.dcut2*d_2), name="or")
        slice_energies = tf.compat.v1.where_v2(tf.math.is_nan(uljcc), _tf_zero, uljcc, name="where_nan")
        slice_energies = tf.compat.v1.where_v2(r < (utility.dcut * d), slice_energies, _tf_zero, name="where_dcut")
        # filtered = tf.compat.v1.debugging.check_numerics(filtered, message="filtered lj forces")
        # print("slice_forces", slice_forces)
        return common.my_tf_round(0.5*tf.math.reduce_sum(tf.math.reduce_sum(slice_energies, axis=1)), 6)

def _left_wall_lj_energy(ion_dict, simul_box):
    """
    left wall
    ion interacting with left wall directly (self, closest)
    """
    with tf.name_scope("left_wall_lj_energy"):
        mask = ion_dict[interface.ion_pos_str][:, -1] < ((-0.5 * simul_box.lz) + ion_dict[interface.ion_diameters_str])  # TODO: remove this mask if not cause of sim error
        dummy_mult = tf.constant([1, 1, 0], name="dummy_mult_left", dtype=common.tf_dtype)
        dummy_pos = ion_dict[interface.ion_pos_str] * dummy_mult
        # TODO!: replace - 0.5 with 0.5* diameter for correctness
        dummy_add = tf.constant([0, 0, (-0.5 * simul_box.lz)], name="dummy_add_left", dtype=common.tf_dtype)
        dummy_pos = dummy_pos + dummy_add
        distances = ion_dict[interface.ion_pos_str] - dummy_pos
        r2 = common.magnitude_squared(distances, axis=1, keepdims=True)  # keep 1th dimension to match up with distances later
        diam_2 = tf.math.pow(ion_dict[interface.ion_diameters_str] * 0.5, 2.0, name="diam_2_pow")[:, tf.newaxis]  # add new dimension to match up with distances later
        d_r_6 = tf.math.pow(diam_2, 3.0, name="diam_6_pow") / tf.math.pow(r2, 3.0, name="r_6_pow")  # magnitude is alread "squared" so only need N/2 power
        ulj = 4 * utility.elj * d_r_6 * (d_r_6 - 1) + utility.elj
        ulj_masked = tf.compat.v1.where_v2(r2 < (diam_2 * utility.dcut2), ulj, _tf_zero, name="where_d_cut")
        return common.my_tf_round(tf.math.reduce_sum(tf.compat.v1.where_v2(mask[:, tf.newaxis], ulj_masked, _tf_zero, name="lj_wall_bulk_cutoff")),6)


def _right_wall_lj_energy(ion_dict, simul_box):
    """
    right wall
    ion interacting with right wall directly (self, closest)
    """
    with tf.name_scope("right_wall_lj_energy"):
        mask = ion_dict[interface.ion_pos_str][:, -1] > ((0.5 * simul_box.lz) - ion_dict[interface.ion_diameters_str])  # TODO: remove this mask if not cause of sim error
        dummy_mult = tf.constant([1, 1, 0], name="dummy_mult_right", dtype=common.tf_dtype)
        dummy_pos = ion_dict[interface.ion_pos_str] * dummy_mult
        # TODO!: replace + 0.5 with 0.5* diameter for correctness
        dummy_add = tf.constant([0, 0, (0.5 * simul_box.lz)], name="dummy_add_right", dtype=common.tf_dtype)
        dummy_pos = dummy_pos + dummy_add
        distances = ion_dict[interface.ion_pos_str] - dummy_pos
        r2 = common.magnitude_squared(distances, axis=1, keepdims=True)  # keep 1th dimension to match up with distances later
        diam_2 = tf.math.pow(ion_dict[interface.ion_diameters_str] * 0.5, 2.0, name="diam_2_pow")[:, tf.newaxis]  # add new dimension to match up with distances later
        d_r_6 = tf.math.pow(diam_2, 3.0, name="diam_6_pow") / tf.math.pow(r2, 3.0, name="r_6_pow")  # magnitude is alread "squared" so only need N/2 power
        ulj = 4 * utility.elj * d_r_6 * (d_r_6 - 1) + utility.elj
        d_cut = tf.compat.v1.where_v2(r2 < (diam_2 * utility.dcut2), ulj, _tf_zero, name="where_d_cut")
        return common.my_tf_round(tf.reduce_sum(tf.compat.v1.where_v2(mask[:, tf.newaxis], d_cut, _tf_zero, name="lj_wall_bulk_cutoff")),6)

def _right_wall_columb_energy(ion_dict, simul_box):
    """
    coulomb interaction ion-rightwall
    """
    with tf.name_scope("electrostatic_right_wall_energy"):
        return _electrostatic_wall_energy(simul_box, ion_dict, simul_box.tf_right_plane)


def _left_wall_columb_energy(ion_dict, simul_box):
    """
    coulomb interaction ion-leftwall
    """
    with tf.name_scope("electrostatic_left_wall_energy"):
        return _electrostatic_wall_energy(simul_box, ion_dict, simul_box.tf_left_plane)

def _electrostatic_wall_energy(simul_box, ion_dict, wall_dictionary):
    with tf.name_scope("electrostatic_wall_energy"):
        wall_distances = common.wrap_vectorize(fn=lambda atom_pos: atom_pos - wall_dictionary["posvec"], elems=ion_dict[interface.ion_pos_str])
        wall_z_dist = wall_distances[:, :, -1]  # get z-axis value
        abs_z_distances = tf.math.abs(wall_z_dist)
        # factor = tf.compat.v1.where_v2(wall_z_dist >= 0.0, _tf_one, _tf_neg_one, name="where_factor")
        r1 = tf.math.sqrt(0.5 + ((wall_z_dist / simul_box.lx) * (wall_z_dist / simul_box.lx)))
        r2 = tf.math.sqrt(0.25 + ((wall_z_dist / simul_box.lx) * (wall_z_dist / simul_box.lx)))
        # condition = tf.equal(wall_z_dist, 0)
        # r1 = tf.compat.v1.where_v2(condition, wall_z_dist, r1, name="r1_cleanup")
        # r2 = tf.compat.v1.where_v2(condition, wall_z_dist, r2, name="r2_cleanup")
        E_z = 4 * tf.math.atan(4 * abs_z_distances * r1 / simul_box.lx)
        fcsh_z = 4 * simul_box.lx * tf.math.log((0.5 + r1) / r2) - abs_z_distances * (2 * utility.pi - E_z)
        fcsh_inf = -2 * utility.pi * abs_z_distances
        ion_one_over_ep = 1 / ion_dict[interface.ion_epsilon_str]  # 1 / ion[i].epsilon
        wall_one_over_ep = 1 / wall_dictionary["epsilon"]  # 1 / wall_dummy.epsilon
        q_over_lx_sq = wall_dictionary["q"] / (simul_box.lx * simul_box.lx)
        vec_one_over_ep = common.wrap_vectorize(fn=lambda epsilon_j: epsilon_j + wall_one_over_ep, elems=ion_one_over_ep)
        # vec_q_over_lx_sq = common.wrap_vectorize(fn=lambda q_j: q_j * ion_dict[interface.ion_charges_str], elems=q_over_lx_sq)
        vec_q_over_lx_sq = common.wrap_vectorize(fn=lambda q_j: q_j * q_over_lx_sq, elems=ion_dict[interface.ion_charges_str])  # ion[i].q * (wall_dummy.q / (box.lx * box.lx))
        fqq_csh_ion = vec_q_over_lx_sq * 0.5 * vec_one_over_ep * (fcsh_inf - fcsh_z)
        fqq_csh = tf.math.reduce_sum(fqq_csh_ion, axis=1, keepdims=True)

        wrapped_distances = common.wrap_distances_on_edges(simul_box, wall_distances)
        # r = tf.norm(wrapped_distances, ord='euclidean', axis=2, keepdims=True)
        r = common.magnitude(wrapped_distances, axis=2, keepdims=True)
        # r3 = tf.math.pow(r, 3)
        vec_q_mul = common.wrap_vectorize(fn=lambda q_j: wall_dictionary["q"] * q_j,elems=ion_dict[interface.ion_charges_str])
        # a = _zero_nans(wrapped_distances * ((-1.0) / r3))  # r3 can have zeroes in it, so remove the nans that come from div by zero
        b = (0.5 * vec_q_mul * vec_one_over_ep)[:, :, tf.newaxis]
        fqq_ion = _zero_nans(b / r)
        fqq = tf.math.reduce_sum(fqq_ion, axis=1, keepdims=True)
        # h1 = tf.math.reduce_sum(a * b[:, :, tf.newaxis], axis=1, keepdims=False,
        #                         name="sum_a_times_b")  # TODO: remove need for newaxis here  #-------------->>>>> change axis here to 0
        # fqq_xy = fqq[:, 0:2]  # TODO: replace this junk with better impl
        # c = fqq[:, 2:3] + fqq_csh
        # con = tf.concat(values=[fqq_xy, c], axis=1, name="x_y_and_c_concatenate")
        return common.my_tf_round(tf.math.reduce_sum(fqq + fqq_csh) * utility.scalefactor, 6)


#TF version
def kinetic_energy(ion_dict):
    with tf.name_scope("kinetic_energy"):
        m = tf.norm(ion_dict[velocities.ion_vel_str],ord='euclidean', axis=1, keepdims=False)
        # print("\n M in tf:", m.eval(session=tf.compat.v1.Session()))
        # out_m = tf.Print(m,[m[0],m[1]],"tf vel")
        ke_particle = 0.5 * ion_dict[interface.ion_masses_str] * m * m   #common.magnitude_squared(ion_dict[velocities.ion_vel_str], axis=1)
        # out_ke = tf.Print(ke_particle, [ke_particle], "ke in tf")
        ke_sum = tf.reduce_sum(ke_particle, keepdims=False)
        # ke_sum_out = tf.Print(ke_sum, [ke_sum],"ke_sum")
        return common.my_tf_round(ke_sum, 6)

#numpy version
def np_kinetic_energy(ion_dict):
    m = common.magnitude_np(ion_dict[velocities.ion_vel_str], axis=1)
    # m = np.linalg.norm(ion_dict[velocities.ion_vel_str])
    # print("\n vel in np:", m[0],m[1])
    ke = 0.5 * ion_dict[interface.ion_masses_str] * m * m
    return np.sum(ke, keepdims=False)

def bath_kinetic_energy(real_bath):
    ke = thermostat.bath_kinetic_energy(real_bath)
    return ke #tf.reduce_sum(ke, axis=0)

def bath_potential_energy(real_bath):
    pe = thermostat.bath_potential_energy(real_bath)
    return pe

def energy_functional(box, charge_meshpoint, ion_dict):
    coulomb_rightwall = tf.cast(0.0, tf.float64)
    coulomb_leftwall = tf.cast(0.0, tf.float64)
    if charge_meshpoint != 0.0:
        coulomb_rightwall = _right_wall_columb_energy(ion_dict, box)
        coulomb_leftwall = _left_wall_columb_energy(ion_dict, box)
    # print("\n coulumb:",tf.reduce_sum(ion_energy(ion_dict, box)).eval(session=tf.compat.v1.Session()), " lj-ion-ion:", tf.reduce_sum(_lj_energy(ion_dict, box)).eval(session=tf.compat.v1.Session())," lj-left-wall:",tf.reduce_sum(_left_wall_lj_energy(ion_dict, box)).eval(session=tf.compat.v1.Session()))
    totalpotential = ion_energy(ion_dict, box) + _lj_energy(ion_dict, box) + _left_wall_lj_energy(ion_dict, box) + _right_wall_lj_energy(ion_dict, box) + coulomb_rightwall + coulomb_leftwall
    #totalpotential = tf.reduce_sum(potential)
    total_electrostatics_walls = box.electrostatics_between_walls()
    totalpotential += total_electrostatics_walls

    return totalpotential  #- (totalpotential%0.0001)

