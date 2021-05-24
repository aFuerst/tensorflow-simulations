import tensorflow as tf
import numpy as np
import utility, interface, common
import os

class Bin:
    def __init__(self):
        volume = None
        width = None
        lower = 0.0
        upper = 0.0
        midpoint = 0.0
        # number_of_bins = None

        # pos_bin_density_records = []
        # neg_bin_density_records = []

    def make_bins(self, box, set_bin_width, ion_diam):
        bins = []
        global number_of_bins
        number_of_bins = int(box.lz / set_bin_width)
        print("number of bins:", number_of_bins, " box.lz:", box.lz," set_bin_width:", set_bin_width," utility.unitlength:",utility.unitlength)
        bin_width = (box.lz / number_of_bins)  # To make discretization of bins symmetric, we recalculate the bin_width
        number_of_bins += 2
        bin_volume = (bin_width * box.lx * box.ly) * (utility.unitlength * utility.unitlength * utility.unitlength) * 0.6022
        f_listbin = open(os.path.join(utility.root_path, "listbin.dat"), 'a')
        for bin_num in range(0, number_of_bins):
            bin = Bin()
            bin.width = bin_width
            bin.volume = bin_volume
            bin.lower = 0.5 * (-box.lz) + bin_num * bin.width
            bin.higher = 0.5 * (-box.lz) + (bin_num + 1) * bin.width
            bin.midpoint = 0.5*(bin.lower + bin.higher)
            f_listbin.write(str(bin.width) + "\t" + str(bin.volume) + "\t" + str(bin.lower) + "\t" + str(bin.higher) + "\t" + str(bin.midpoint)+"\n")
            bins.append(bin)
        # This is to get contact point densities
        leftContact = -0.5 * box.lz + 0.5 * ion_diam - 0.5 * bins[0].width
        rightContact = 0.5 * box.lz - 0.5 * ion_diam - 0.5 * bins[0].width
        # print("leftcontact:", leftContact," rightcontact:", rightContact)
        bins[len(bins) - 1].lower = leftContact
        bins[len(bins) - 2].lower = rightContact
        bins[len(bins) - 1].higher = leftContact + bin_width
        bins[len(bins) - 2].higher = rightContact + bin_width
        bins[len(bins) - 1].midpoint = 0.5 * (bins[len(bins) - 1].lower + bins[len(bins) - 1].higher)
        bins[len(bins) - 2].midpoint = 0.5 * (bins[len(bins) - 2].lower + bins[len(bins) - 2].higher)
        return bins #{"bin_width":bin_width, "bin_midpoints":common.py_array_to_np(bin_midpoints), "bin_volume":bin_volume, "number_of_bins":number_of_bins, "bin_arr":[]}

    def bin_ions(self, box, ion_dict, bins):
        with tf.name_scope("calc_bin_density"):
            charge_filter = tf.math.greater(ion_dict[interface.ion_valency_str], 0)
            neg_charge_filter = tf.math.logical_not(charge_filter)
            z_pos = ion_dict[interface.ion_pos_str][:, -1]  # get z-axis value
            bin_nums = tf.dtypes.cast((z_pos + 0.5*box.lz) / bins[0].width, tf.int32)
            out_bin_nums = tf.Print(bin_nums, [bins[0].volume, box.lz, bins[0].width, z_pos[0], tf.dtypes.cast((z_pos[0] + 0.5*box.lz) / bins[0].width, tf.int32)], " bin_nums")
            pos_bin_density = tf.math.bincount(tf.compat.v1.boolean_mask(out_bin_nums, charge_filter), minlength=len(bins), maxlength=len(bins), dtype=common.tf_dtype) / bins[0].volume
            neg_bin_density = tf.math.bincount(tf.compat.v1.boolean_mask(bin_nums, neg_charge_filter), minlength=len(bins), maxlength=len(bins), dtype=common.tf_dtype) / bins[0].volume
            # out_pos_bin_density = tf.Print(pos_bin_density, [pos_bin_density[9], neg_bin_density[9]], "out_pos_bin_density")
            return pos_bin_density, neg_bin_density

    def record_densities(self, iter, pos_bin_density, neg_bin_density, no_samples, bins, mean_pos_density, mean_sq_pos_density, mean_neg_density, mean_sq_neg_density, simul_box, ion_dict):
        # pos_bin_density_records.append(pos_bin_density)
        # neg_bin_density_records.append(neg_bin_density)
        # print("pos_bin_density:", pos_bin_density)
        # (pos_bin_density, neg_bin_density) = bin_ions(simul_box, ion_dict, bins)
        for i in range(0, len(mean_pos_density)):
            mean_pos_density[i] += pos_bin_density[i]
        # out_mean_pos_density = tf.Print(mean_pos_density, [pos_bin_density[9], pos_bin_density[10]], "mean_pos_density")
        for i in range(0, len(mean_neg_density)):
            mean_neg_density[i] += neg_bin_density[i]
        for i in range(0, len(mean_sq_pos_density)):
            mean_sq_pos_density[i] += pos_bin_density[i]*pos_bin_density[i]
        for i in range(0, len(mean_sq_neg_density)):
            mean_sq_neg_density[i] += neg_bin_density[i]*neg_bin_density[i]
        path = utility.root_path
        outdenp = open(os.path.join(path,"_z+_den-{}.dat".format(iter)), 'w')
        outdenn = open(os.path.join(path,"_z-_den-{}.dat".format(iter)), 'w')
        # print("no sample:", no_samples)
        for b in range(0, len(bins)):
            print("midpoint of bin:",b,"is:",bins[b].midpoint)
            outdenp.write(str(bins[b].midpoint*utility.unitlength)+"\t"+str(mean_pos_density[b]/no_samples)+"\n")
            outdenn.write(str(bins[b].midpoint*utility.unitlength)+"\t"+str(mean_neg_density[b]/no_samples)+"\n")
        return mean_pos_density, mean_sq_pos_density, mean_neg_density, mean_sq_neg_density
