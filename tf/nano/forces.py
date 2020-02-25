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
        edges = tf.constant([sumil_box.lx, sumil_box.ly, 0], name="box_edges", dtype=common.tf_dtype)
        edges_half = tf.constant([sumil_box.lx/2, sumil_box.ly/2, 0], name="box_edges_half", dtype=common.tf_dtype)
        neg_edges_half = tf.constant([-sumil_box.lx/2, -sumil_box.ly/2, 0], name="box_edges_half", dtype=common.tf_dtype)
        wrapped_distances = tf.compat.v1.where_v2(distances > edges_half, distances - edges, distances, name="where_edges_half")
        return tf.compat.v1.where_v2(wrapped_distances < neg_edges_half, wrapped_distances + edges, wrapped_distances, name="where_neg_edges_half")

_tf_zero = tf.constant(0, name="nan_zeroer", dtype=common.tf_dtype)

def _zero_nans(tensor):
    """
    Replaces all nans in the given tensor with 0s
    """
    with tf.name_scope("zero_nans"):
        return tf.compat.v1.where_v2(tf.math.is_nan(tensor), _tf_zero, tensor, name="zero_nans_where")

@tf.function
def _particle_electrostatic_force(sumil_box, ion_dict):
    """
    force on the particles (electrostatic)
    parallel calculation of forces (uniform case)
    """
    with tf.name_scope("left_wall_lj_force"):
        distances = tf.compat.v1.vectorized_map(fn=lambda atom_pos: ion_dict["ion_pos"] - atom_pos, elems=ion_dict["ion_pos"])
        z_distances = distances[:, :, -1] # get z-axis value
        abs_z_distances = tf.math.abs(z_distances)
        r1 = tf.math.sqrt(0.5 + (z_distances / sumil_box.lx) * (z_distances / sumil_box.lx))
        r2 = tf.math.sqrt(0.25 + (z_distances / sumil_box.lx) * (z_distances / sumil_box.lx))
        E_z = 4 * tf.math.atan(4 * abs_z_distances * r1 / sumil_box.lx)

        factor = tf.compat.v1.where_v2(z_distances >= 0.0, common.py_array_to_np(1.0), common.py_array_to_np(-1.0), name="where_factor")
        hcsh = (4 / sumil_box.lx) * (1 / (r1 * (0.5 + r1)) - 1 / (r2 * r2)) * z_distances + factor * E_z + \
                       16 * abs_z_distances * (sumil_box.lx / (sumil_box.lx * sumil_box.lx + 16 * z_distances * z_distances * r1 * r1)) * \
                       (abs_z_distances * z_distances / (sumil_box.lx * sumil_box.lx * r1) + factor * r1) # MATHEMATICAL

        #h1.z = h1.z + 2 * ion[i].q * (ion[j].q / (box.lx * box.lx)) * 0.5 * (1 / ion[i].epsilon + 1 / ion[j].epsilon) * hcsh
        one_over_ep = 1 / ion_dict["ion_diconst"]
        q_over_lx_sq = ion_dict["ion_charges"] / (sumil_box.lx * sumil_box.lx)
        vec_one_over_ep = tf.compat.v1.vectorized_map(fn=lambda disconst_j: one_over_ep + disconst_j, elems=one_over_ep)
        vec_q_over_lx_sq = tf.compat.v1.vectorized_map(fn=lambda charges_j: ion_dict["ion_charges"] * charges_j, elems=q_over_lx_sq)

        h1_z = 2 * vec_q_over_lx_sq * 0.5 * (vec_one_over_ep) * hcsh
        h1_z = tf.math.reduce_sum(h1_z, axis=1, keepdims=True)
        
        # h1 =h1+ ((temp_vec ^ ((-1.0) / r3)) ^ ((-0.5) * ion[i].q * ion[j].q * (1 / ion[i].epsilon + 1 / ion[j].epsilon)));
        wrapped_distances = _wrap_distances_on_edges(sumil_box, distances)
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
def _right_wall_lj_force():
    """
    interaction with the right plane hard wall
    make a dummy particle with the same diameter as the ion and touching right of the right wall s. t. it is closest to the ion
    """
    pass

@tf.function
def _particle_lj_force(sumil_box, ion_dict):
    """
    excluded volume interactions given by purely repulsive LJ
    ion-ion
    """
    with tf.name_scope("particle_lj_force"):
        distances = tf.compat.v1.vectorized_map(fn=lambda atom_pos: ion_dict["ion_pos"] - atom_pos, elems=ion_dict["ion_pos"])
        diams_sum = tf.compat.v1.vectorized_map(fn=lambda atom_pos: ion_dict["ion_diameters"] + atom_pos, elems=ion_dict["ion_diameters"]) * 0.5
        diams_sum = diams_sum[:,:,tf.newaxis] # add third dimension to match with wrapped_distances and mag_squared later
        wrapped_distances = _wrap_distances_on_edges(sumil_box, distances)
        mag_squared = common.magnitude_squared(wrapped_distances, keepdims=True) # keep third dimension to match with wrapped_distances
        #TODO: this calculation is different than original LJ, ask why
        diam_2 = tf.math.pow(diams_sum, 2.0, name="square_diam_diff")
        d_six = tf.math.pow(diams_sum, 6.0, name="diam_6_pow") / tf.math.pow(mag_squared, 3.0, name="mag_6_pow") # magnitude is alread "squared" so only need N/2 power
        d_twelve = tf.math.pow(diams_sum, 12.0, name="diam_12_pow") / tf.math.pow(mag_squared, 6.0, name="mag_12_pow")
        slice_forces = wrapped_distances * (48.0 * 1.0 * (((d_twelve - 0.5 * d_six) * (1.0/mag_squared)))) # elj = 1.0
        # handle case distances pos - atom_pos == 0, causing inf and nan to appear in that position
        filter = tf.math.logical_or(tf.math.is_nan(slice_forces), mag_squared < (diam_2*utility.dcut2), name="or")
        filtered = tf.compat.v1.where_v2(filter, _tf_zero, slice_forces, name="where_or")
        return tf.math.reduce_sum(filtered, axis=0)
        
@tf.function
def _left_wall_lj_force(sumil_box, ion_dict):
    """
    ion-box
    interaction with the left plane hard wall
    make a dummy particle with the same diameter as the ion and touching left of the left wall s. t. it is closest to the ion
    """
    """
    for (i = lowerBound; i <= upperBound; i++) {
        flj = VECTOR3D(0, 0, 0);
        if (ion[i].posvec.z > 0.5 * box.lz - ion[i].diameter)  // avoiding calculating interactions between right wall and ions in bulk. 
        {
            dummy = PARTICLE(0, 0, 0, 0, 0, box.eout, VECTOR3D(ion[i].posvec.x, ion[i].posvec.y, 0.5 * box.lz), box.lx, box.ly, box.lz);
            r_vec = ion[i].posvec - dummy.posvec;
            r2 = r_vec.GetMagnitudeSquared();
            d = 0.5 * ion[i].diameter;
            d2 = d * d;
            elj = 1.0;
            if (r2 < dcut2 * d2) {
                r6 = r2 * r2 * r2;
                r12 = r6 * r6;
                d6 = d2 * d2 * d2;
                d12 = d6 * d6;
                flj = r_vec ^ (48 * elj * ((d12 / r12) - 0.5 * (d6 / r6)) * (1 / r2));
            }
        }
        lj_ion_rightdummy[i - lowerBound] = flj;
    }
    """
    pass

@tf.function
def _electrostatic_right_wall_force():
    """
    ion interacting with discretized right wall
    electrostatic between ion and rightwall
    """
    """
    for (i = lowerBound; i <= upperBound; i++)
    {
      h1_rightwall = VECTOR3D(0, 0, 0);
      for (k = 0; k < box.rightplane.size(); k++)
      {
        wall_dummy = PARTICLE(0, 0, valency_counterion * -1, charge_meshpoint * 1.0, 0, box.eout, VECTOR3D(box.rightplane[k].posvec.x, box.rightplane[k].posvec.y, box.rightplane[k].posvec.z), box.lx, box.ly, box.lz);

        dz_rightwall = ion[i].posvec.z - wall_dummy.posvec.z;
        if (dz_rightwall >= 0) factor = 1;
        else factor = -1;
        r1_rightwall = sqrt(0.5 + (dz_rightwall / box.lx) * (dz_rightwall / box.lx));
        r2_rightwall = sqrt(0.25 + (dz_rightwall / box.lx) * (dz_rightwall / box.lx));
        E_z_rightwall = 4 * atan(4 * fabs(dz_rightwall) * r1_rightwall / box.lx);
        hcsh_rightwall = (4 / box.lx) * (1 / (r1_rightwall * (0.5 + r1_rightwall)) - 1 / (r2_rightwall * r2_rightwall)) * dz_rightwall + factor * E_z_rightwall +
               16 * fabs(dz_rightwall) * (box.lx / (box.lx * box.lx + 16 * dz_rightwall * dz_rightwall * r1_rightwall * r1_rightwall)) *
               (fabs(dz_rightwall) * dz_rightwall / (box.lx * box.lx * r1_rightwall) + factor * r1_rightwall);

        h1_rightwall.z = h1_rightwall.z + 2 * ion[i].q * (wall_dummy.q / (box.lx * box.lx)) * 0.5 * (1 / ion[i].epsilon + 1 / wall_dummy.epsilon) * hcsh_rightwall;

        temp_vec_rightwall = ion[i].posvec - wall_dummy.posvec;
        if (temp_vec_rightwall.x > box.lx / 2) temp_vec_rightwall.x -= box.lx;
        if (temp_vec_rightwall.x < -box.lx / 2) temp_vec_rightwall.x += box.lx;
        if (temp_vec_rightwall.y > box.ly / 2) temp_vec_rightwall.y -= box.ly;
        if (temp_vec_rightwall.y < -box.ly / 2) temp_vec_rightwall.y += box.ly;
        r_rightwall = temp_vec_rightwall.GetMagnitude();
        r3_rightwall = r_rightwall * r_rightwall * r_rightwall;
        h1_rightwall = h1_rightwall+ ((temp_vec_rightwall ^ ((-1.0) / r3_rightwall)) ^ ((-0.5) * ion[i].q * wall_dummy.q * (1 / ion[i].epsilon + 1 / wall_dummy.epsilon)));
       }
       coulomb_rightwallForce[i - lowerBound] = ((h1_rightwall) ^ (scalefactor));
    }
    """
    pass

@tf.function
def _electrostatic_left_wall_force():
    """
    ion interacting with discretized left wall
    electrostatic between ion and left wall
    """
    """
    for (i = lowerBound; i <= upperBound; i++)
    {
      h1_leftwall = VECTOR3D(0, 0, 0);
      for (m = 0; m < box.leftplane.size(); m++)
      {
        wall_dummy = PARTICLE(0, 0, valency_counterion * -1, charge_meshpoint * 1.0, 0, box.eout, VECTOR3D(box.leftplane[m].posvec.x, box.leftplane[m].posvec.y, box.leftplane[m].posvec.z), box.lx, box.ly, box.lz);
		  
        dz_leftwall = ion[i].posvec.z - wall_dummy.posvec.z;
        if (dz_leftwall >= 0) factor = 1;
        else factor = -1;
        r1_leftwall = sqrt(0.5 + (dz_leftwall / box.lx) * (dz_leftwall / box.lx));
        r2_leftwall = sqrt(0.25 + (dz_leftwall / box.lx) * (dz_leftwall / box.lx));
        E_z_leftwall  = 4 * atan(4 * fabs(dz_leftwall) * r1_leftwall / box.lx);
        hcsh_leftwall = (4 / box.lx) * (1 / (r1_leftwall * (0.5 + r1_leftwall)) - 1 / (r2_leftwall * r2_leftwall)) * dz_leftwall + factor * E_z_leftwall  +
               16 * fabs(dz_leftwall) * (box.lx / (box.lx * box.lx + 16 * dz_leftwall * dz_leftwall * r1_leftwall * r1_leftwall)) *
               (fabs(dz_leftwall) * dz_leftwall / (box.lx * box.lx * r1_leftwall) + factor * r1_leftwall);

        h1_leftwall.z = h1_leftwall.z + 2 * ion[i].q * (wall_dummy.q / (box.lx * box.lx)) * 0.5 * (1 / ion[i].epsilon + 1 / wall_dummy.epsilon) * hcsh_leftwall;

        temp_vec_leftwall = ion[i].posvec - wall_dummy.posvec;
        if (temp_vec_leftwall.x > box.lx / 2) temp_vec_leftwall.x -= box.lx;
        if (temp_vec_leftwall.x < -box.lx / 2) temp_vec_leftwall.x += box.lx;
        if (temp_vec_leftwall.y > box.ly / 2) temp_vec_leftwall.y -= box.ly;
        if (temp_vec_leftwall.y < -box.ly / 2) temp_vec_leftwall.y += box.ly;
        r_leftwall = temp_vec_leftwall.GetMagnitude();
        r3_leftwall = r_leftwall * r_leftwall * r_leftwall;
        h1_leftwall = h1_leftwall + ((temp_vec_leftwall ^ ((-1.0) / r3_leftwall)) ^ ((-0.5) * ion[i].q * wall_dummy.q * (1 / ion[i].epsilon + 1 / wall_dummy.epsilon)));
      }
      coulomb_leftwallForce[i - lowerBound] = ((h1_leftwall) ^ (scalefactor));
    }
    """
    pass

@tf.function
def for_md_calculate_force(sumil_box, thermostats, ion_dict, bins, charge_meshpoint: float, valency_counterion: int):
    pass
    
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
    charge_density = 0.0
    valency_counterion = 1
    charge_meshpoint = (charge_density * surface_area) / (utility.unitcharge * number_meshpoints) # in unit of electron charge
    total_surface_charge = charge_meshpoint * number_meshpoints # in unit of electron charge
    counterions =  2.0 * (int(abs(total_surface_charge)/valency_counterion)) # there are two charged surfaces, we multiply the counter ions by two
    print(counterions)
    sumil_box = interface.Interface(salt_conc_in=0.5, salt_conc_out=0, salt_valency_in=1, salt_valency_out=1, bx=bx, by=by, bz=bz, initial_ein=1, initial_eout=1)
    ion_dict = sumil_box.put_saltions_inside(pz=1, nz=-1, concentration=0.5, positive_diameter_in=0.474, negative_diameter_in=0.627, \
                            counterions=counterions, valency_counterion=1, counterion_diameter_in=0.474, bigger_ion_diameter=0.627)

    tf_ion_real, tf_ion_place = common.make_tf_versions_of_dict(ion_dict)
    sess.run(tf.compat.v1.global_variables_initializer())
    feed = {}
    for key in ion_dict.keys():
        feed[tf_ion_place[key]] = ion_dict[key]
    pef = _particle_electrostatic_force(sumil_box, tf_ion_real)
    check_op = tf.compat.v1.add_check_numerics_ops()
    print("\n\n","pef",pef)
    pef, _ = sess.run(fetches=[pef,check_op], feed_dict=feed)
    print(pef)
    if(pef.shape != (424,3)):
        raise Exception("bad shape {}".format(pef.shape))
    print("\n\n")
    pef = _particle_lj_force(sumil_box, tf_ion_real)
    print(pef)
    print(pef.eval(session=sess))
    if(pef.shape != (424,3)):
        raise Exception("bad shape {}".format(pef.shape))
