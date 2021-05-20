import tensorflow as tf
import numpy as np
import utility, interface, common
import os


bin_volume = None
bin_width = None
number_of_bins = None
pos_bin_density_records = []
neg_bin_density_records = []

def make_bins(box, set_bin_width):
    global bin_volume
    global bin_width
    global number_of_bins
    bin_midpoints = []
    number_of_bins = int(box.lz / set_bin_width)
    print("number of bins:", number_of_bins, " box.lz:", box.lz," set_bin_width:", set_bin_width)
    bin_width = (box.lz / number_of_bins)  # To make discretization of bins symmetric, we recalculate the bin_width
    bin_volume = (bin_width * box.lx * box.ly) * (utility.unitlength * utility.unitlength * utility.unitlength) * 0.6022
    number_of_bins += 2
    for bin_num in range(0, number_of_bins):
        lower = 0.5 * (-box.lz) + bin_num * bin_width
        higher = 0.5 * (-box.lz) + (bin_num + 1) * bin_width
        bin_midpoints.append(0.5*(lower + higher))

    return {"bin_width":bin_width, "bin_midpoints":common.py_array_to_np(bin_midpoints), "bin_volume":bin_volume, "number_of_bins":number_of_bins, "bin_arr":[]}

# def bin_ions(box, ion_dict, bins):
#     for i in range(0, bins["number_of_bins"]):
#         bins["bin_arr"][i] = 0
#     # for j in range(0, ion_dict.)
#     r_vec = 0.5 * box.lz + ion_dict[interface.ion_pos_str]
#     print("==== r_vec is:", r_vec)

def bin_ions(box, ion_dict, bins):
    with tf.name_scope("calc_bin_density"):
        charge_filter = tf.math.greater(ion_dict[interface.ion_valency_str], 0)
        neg_charge_filter = tf.math.logical_not(charge_filter)
        z_pos = ion_dict[interface.ion_pos_str][:, -1]  # get z-axis value
        bin_nums = tf.dtypes.cast((z_pos + 0.5*box.lz) / bins["bin_width"], tf.int32)
        # out_bin_nums = tf.Print(bin_nums, [bin_volume, box.lz, bins["bin_width"], z_pos[0], tf.dtypes.cast((z_pos[0] + 0.5*box.lz) / bins["bin_width"], tf.int32)], " bin_nums")
        pos_bin_density = tf.math.bincount(tf.compat.v1.boolean_mask(bin_nums, charge_filter), minlength=number_of_bins, maxlength=number_of_bins, dtype=common.tf_dtype) / bin_volume
        neg_bin_density = tf.math.bincount(tf.compat.v1.boolean_mask(bin_nums, neg_charge_filter), minlength=number_of_bins, maxlength=number_of_bins, dtype=common.tf_dtype) / bin_volume
        # out_pos_bin_density = tf.Print(pos_bin_density, [pos_bin_density[10], neg_bin_density[10]], "out_pos_bin_density")
        return pos_bin_density, neg_bin_density

def record_densities(iter, pos_bin_density, neg_bin_density, no_samples, bins, mean_pos_density, mean_sq_pos_density, mean_neg_density, mean_sq_neg_density, simul_box, ion_dict):
    pos_bin_density_records.append(pos_bin_density)
    neg_bin_density_records.append(neg_bin_density)
    # print("pos_bin_density:", pos_bin_density)
    # (pos_bin_density, neg_bin_density) = bin_ions(simul_box, ion_dict, bins)
    for i in range(0, len(mean_pos_density)):
        mean_pos_density[i] += pos_bin_density[i]
    for i in range(0, len(mean_neg_density)):
        mean_neg_density[i] += neg_bin_density[i]
    for i in range(0, len(mean_sq_pos_density)):
        mean_sq_pos_density[i] += pos_bin_density[i]*pos_bin_density[i]
    for i in range(0, len(mean_sq_neg_density)):
        mean_sq_neg_density[i] += neg_bin_density[i]*neg_bin_density[i]
    path = utility.root_path
    outdenp = open(os.path.join(path,"_z+_den-{}.dat".format(iter)), 'w')
    outdenn = open(os.path.join(path,"_z-_den-{}.dat".format(iter)), 'w')
    for b in range(0, bins["number_of_bins"]):
        outdenp.write(str(utility.unitlength*bins["bin_midpoints"][b])+"\t"+str(mean_pos_density[b]/no_samples)+"\n")
        outdenn.write(str(bins["bin_midpoints"][b]*utility.unitlength)+"\t"+str(mean_neg_density[b]/no_samples)+"\n")

    return mean_pos_density, mean_sq_pos_density, mean_neg_density, mean_sq_neg_density


# if __name__ == "__main__":
#     tf.compat.v1.random.set_random_seed(0)
#     from tensorflow_manip import silence, toggle_cpu
#     silence()
#     utility.unitlength = 1
#     bz = 3
#     utility.scalefactor = utility.epsilon_water * utility.lB_water / utility.unitlength
#     ion_dict = {
#                     interface.ion_pos_str:tf.random.uniform((50,3), minval=-bz/2, maxval=bz/2),
#                     interface.ion_diameters_str:tf.ones((50,3)),
#                     interface.ion_charges_str:tf.random.uniform((50,), minval=-1, maxval=1)
#                }
#     simul_box = interface.Interface(salt_conc_in=0.5, salt_conc_out=0, salt_valency_in=1, salt_valency_out=1, bx=3, by=3, bz=bz, initial_ein=1, initial_eout=1)
#     make_bins(simul_box, set_bin_width=0.05)
#     print("number_of_bins", number_of_bins)
#     sess = tf.compat.v1.Session()
#     sess.as_default()
#     sess.run(tf.compat.v1.global_variables_initializer())
#
#     p, n = tf_get_ion_bin_density(simul_box, ion_dict)
#     p, n = sess.run((p, n))
#     print("pos", p)
#     print("neg", n)


        #
