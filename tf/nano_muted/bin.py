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
    number_of_bins = int(box.lz / set_bin_width) + 2
    print("number of bins:", number_of_bins, " box.lz:", box.lz," set_bin_width:", set_bin_width)
    bin_width = (box.lz / number_of_bins)  # To make discretization of bins symmetric, we recalculate the bin_width
    bin_volume = (bin_width * box.lx * box.ly) * (utility.unitlength * utility.unitlength * utility.unitlength) * 0.6022
    for bin_num in range(0, number_of_bins):
        lower = 0.5 * (-box.lz) + bin_num * bin_width
        higher = 0.5 * (-box.lz) + (bin_num + 1) * bin_width
        bin_midpoints.append(0.5*(lower + higher))
    return {"bin_width":bin_width, "bin_midpoints":common.py_array_to_np(bin_midpoints), "bin_volume":bin_volume, "number_of_bins":number_of_bins}


def tf_get_ion_bin_density(box, ion_dict, bins):
    charge_filter = tf.math.greater(ion_dict[interface.ion_charges_str], 0)
    neg_charge_filter = tf.math.logical_not(charge_filter)
    z_pos = ion_dict[interface.ion_pos_str][:, -1]  # get z-axis value
    bin_nums = tf.dtypes.cast((z_pos + 0.5*box.lz) / bins["bin_width"], tf.int32)
    print("\n box.lz:", box.lz)
    # charge = ion_dict[interface.ion_charges_str]
    pos = ion_dict[interface.ion_pos_str]
    vel = ion_dict["ion_velocities"]
    force = ion_dict[interface.ion_for_str]

    # # print("\n box.lz:",box.lz*0.5," bin width:", bin_width, " number of bins:", number_of_bins)
    # out_bin_nums = tf.Print(bin_nums, [vel[0], vel[1], pos[0], pos[1], force[0], force[1], ], " force")
    pos_bin_density = tf.math.bincount(tf.compat.v1.boolean_mask(bin_nums, charge_filter), minlength=number_of_bins, maxlength=number_of_bins, dtype=common.tf_dtype) / bin_volume
    neg_bin_density = tf.math.bincount(tf.compat.v1.boolean_mask(bin_nums, neg_charge_filter), minlength=number_of_bins, maxlength=number_of_bins, dtype=common.tf_dtype) / bin_volume
    # print("\n pos_bin_density:", pos_bin_density)
    # out_pos_bin_density = tf.Print(pos_bin_density, [pos_bin_density[9], pos_bin_density[28]],"out_pos_bin_density")
    return pos_bin_density, neg_bin_density

def record_densities(iter, pos_bin_density, neg_bin_density, no_samples, bins, writedensity):
    pos_bin_density_records.append(pos_bin_density)
    neg_bin_density_records.append(neg_bin_density)
    mean_pos_bin_density = np.sum(pos_bin_density_records, axis=0, keepdims=False) / no_samples
    mean_neg_bin_density = np.sum(neg_bin_density_records, axis=0, keepdims=False) / no_samples
    if iter % writedensity==0:
        path = utility.root_path
        outdenp = open(os.path.join(path,"_z+_den-{}.dat".format(iter)), 'w')
        outdenn = open(os.path.join(path,"_z-_den-{}.dat".format(iter)), 'w')
        for b in range(0, len(mean_pos_bin_density)):
            outdenp.write(str(utility.unitlength*bins["bin_midpoints"][b])+"\t"+str(mean_pos_bin_density[b])+"\n")
            outdenn.write(str(bins["bin_midpoints"][b]*utility.unitlength)+"\t"+str(mean_neg_bin_density[b])+"\n")


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