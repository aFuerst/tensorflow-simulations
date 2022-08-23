import tensorflow as tf
import numpy as np
import utility, interface, common
import os
import math

class Bin:
    def __init__(self):
        self.volume = None
        self.width = None
        self.lower = 0.0
        self.upper = 0.0
        self.midpoint = 0.0
        self.mean_pos_density = 0.0
        self.mean_neg_density = 0.0
        self.mean_sq_pos_density = 0.0
        self.mean_sq_neg_density = 0.0

    def make_bins(self, box, set_bin_width, ion_diam):
        bins = []
        global number_of_bins
        number_of_bins = int(box.lz / set_bin_width)

        bin_width = (box.lz / number_of_bins)  # To make discretization of bins symmetric, we recalculate the bin_width
        number_of_bins += 2
        # print("number of bins:", number_of_bins, "\n box.lz:", box.lz, "\n set_bin_width:", set_bin_width,"\n utility.unitlength:", utility.unitlength)
        bin_volume = (bin_width * box.lx * box.ly) * (utility.unitlength * utility.unitlength * utility.unitlength) * 0.6022
        f_listbin = open(os.path.join(utility.root_path, "listbin.dat"), 'a')
        for bin_num in range(0, number_of_bins):
            bin = Bin()
            bin.width = bin_width
            bin.volume = bin_volume
            bin.lower = 0.5 * (-box.lz) + bin_num * bin.width
            bin.higher = 0.5 * (-box.lz) + (bin_num + 1) * bin.width
            bin.midpoint = 0.5*(bin.lower + bin.higher)
            bins.append(bin)
        # This is to get contact point densities
        leftContact = -0.5 * box.lz + 0.5 * ion_diam - 0.5 * bins[0].width
        rightContact = 0.5 * box.lz - 0.5 * ion_diam - 0.5 * bins[0].width
        bins[len(bins) - 1].lower = leftContact
        bins[len(bins) - 2].lower = rightContact
        bins[len(bins) - 1].higher = leftContact + bin_width
        bins[len(bins) - 2].higher = rightContact + bin_width
        bins[len(bins) - 1].midpoint = 0.5 * (bins[len(bins) - 1].lower + bins[len(bins) - 1].higher)
        bins[len(bins) - 2].midpoint = 0.5 * (bins[len(bins) - 2].lower + bins[len(bins) - 2].higher)
        #write listbin here
        for bin_num in range(0, len(bins)):
            f_listbin.write(
                str(bins[bin_num].width) + "\t" + str(bins[bin_num].volume) + "\t" + str(bins[bin_num].lower) + "\t" + str(bins[bin_num].higher) + "\t" + str(
                    bins[bin_num].midpoint) + "\n")
        return bins

    def bin_ions(self, box, ion_dict, bins):
        with tf.name_scope("calc_bin_density"):
            charge_filter = tf.math.greater(ion_dict[interface.ion_valency_str], 0)
            neg_charge_filter = tf.math.logical_not(charge_filter)
            z_pos = ion_dict[interface.ion_pos_str][:, -1]  # get z-axis value
            bin_nums = tf.dtypes.cast((z_pos + 0.5*box.lz) / bins[0].width, tf.int32)
            contact_filter_1 = tf.math.logical_and(tf.math.greater_equal(z_pos, bins[len(bins)-1].lower), tf.math.less(z_pos, bins[len(bins)-1].higher))
            contact_filter_2 = tf.math.logical_and(tf.math.greater_equal(z_pos, bins[len(bins) - 2].lower),
                                                   tf.math.less(z_pos, bins[len(bins) - 2].higher))
            bin_nums = tf.compat.v1.where_v2(contact_filter_1, len(bins) - 1, bin_nums)
            bin_nums = tf.compat.v1.where_v2(contact_filter_2, len(bins) - 2, bin_nums)
            # out_bin_nums = tf.Print(bin_nums, [bin_nums[0], bins[len(bins)-1].lower, bins[len(bins)-1].higher, box.lz, bins[0].width, z_pos[0], tf.dtypes.cast((z_pos[0] + 0.5*box.lz) / bins[0].width, tf.int32)], " bin_nums")
            pos_bin_density = tf.math.bincount(tf.compat.v1.boolean_mask(bin_nums, charge_filter), minlength=len(bins), maxlength=len(bins), dtype=common.tf_dtype) / bins[0].volume
            neg_bin_density = tf.math.bincount(tf.compat.v1.boolean_mask(bin_nums, neg_charge_filter), minlength=len(bins), maxlength=len(bins), dtype=common.tf_dtype) / bins[0].volume
            # out_pos_bin_density = tf.Print(pos_bin_density,[pos_bin_density[9]],"pos_bin_density")
            return pos_bin_density, neg_bin_density

    def record_densities(self, iter, pos_bin_density, neg_bin_density, no_samples, bins, writeDensityFreq):
        for i in range(0, len(bins)):
            bins[i].mean_pos_density += pos_bin_density[i]
            bins[i].mean_neg_density += neg_bin_density[i]
            bins[i].mean_sq_pos_density += pos_bin_density[i]*pos_bin_density[i]
            bins[i].mean_sq_neg_density += neg_bin_density[i]*neg_bin_density[i]
        if iter % writeDensityFreq == 0:
            outdenp = open(os.path.join(utility.root_path,"_z+_den-{}.dat".format(iter)), 'w')
            outdenn = open(os.path.join(utility.root_path,"_z-_den-{}.dat".format(iter)), 'w')
            bins_sorted = sorted(bins, key=lambda x: x.midpoint, reverse=False)
            for b in range(0, len(bins)):
                outdenp.write(str(bins_sorted[b].midpoint*utility.unitlength)+"\t"+str(bins_sorted[b].mean_pos_density/no_samples)+"\n")
                outdenn.write(str(bins_sorted[b].midpoint*utility.unitlength)+"\t"+str(bins_sorted[b].mean_neg_density/no_samples)+"\n")
        return bins

    def average_errorbars_density(self, density_profile_samples, ion_dict_out, simul_box, bins, simul_params):

        # 1. density profile
        bins_sorted = sorted(bins, key=lambda x: x.midpoint, reverse=False)
        positiveion_density_profile = []
        negativeion_density_profile = []
        for b in range(0, len(bins_sorted)):
            positiveion_density_profile.append(bins_sorted[b].mean_pos_density / density_profile_samples)
            negativeion_density_profile.append(bins_sorted[b].mean_neg_density / density_profile_samples)

        # 2. error bars: Standard error is used not std
        p_error_bar = []
        n_error_bar = []
        for b in range(0, len(bins_sorted)):
            p_error_bar.append(np.sqrt(1/density_profile_samples) * np.sqrt(bins_sorted[b].mean_sq_pos_density/density_profile_samples - positiveion_density_profile[b] * positiveion_density_profile[b]))
            n_error_bar.append(np.sqrt(1/density_profile_samples) * np.sqrt(bins_sorted[b].mean_sq_neg_density/density_profile_samples - negativeion_density_profile[b] * negativeion_density_profile[b]))

        # 3. open files
        p_density_profile = "p_density_profile.dat"
        n_density_profile = "n_density_profile.dat"
        f_pos = open(os.path.join(utility.root_path, p_density_profile), 'w')
        f_neg = open(os.path.join(utility.root_path, n_density_profile), 'w')
        # p_density_profile = "p_density_profile.dat"
        # n_density_profile = "n_density_profile.dat"
        # f_pos = open(os.path.join("input/", p_density_profile), 'a')
        # f_neg = open(os.path.join("input/", n_density_profile), 'a')

        # 4. Interpolation to have 150 density profile values always
        z_val = [x.midpoint for x in bins_sorted]
        left = len(z_val) // 2
        right = len(z_val) // 2
        while left-1 >= 0 and right+1 < len(z_val):
            if len(z_val) >= 150:
                break
            positiveion_density_profile.append((positiveion_density_profile[left-1] + positiveion_density_profile[left])/2)
            z_val.append((z_val[left-1] + z_val[left])/2)
            p_error_bar.append((p_error_bar[left-1] + p_error_bar[left])/2)
            negativeion_density_profile.append(
                (negativeion_density_profile[left - 1] + negativeion_density_profile[left]) / 2)
            n_error_bar.append((n_error_bar[left - 1] + n_error_bar[left]) / 2)
            left -= 1

            positiveion_density_profile.append((positiveion_density_profile[right + 1] + positiveion_density_profile[right]) / 2)
            z_val.append((z_val[right + 1] + z_val[right]) / 2)
            p_error_bar.append((p_error_bar[right + 1] + p_error_bar[right]) / 2)
            negativeion_density_profile.append((negativeion_density_profile[right + 1] + negativeion_density_profile[right]) / 2)
            n_error_bar.append((n_error_bar[right + 1] + n_error_bar[right]) / 2)
            right += 1

        # 5. Sort the three lists
        sorted_zips_p = sorted(zip(z_val, positiveion_density_profile, p_error_bar), key=lambda tup: tup[0], reverse=False) #sorted(bins, key=lambda x: x.midpoint, reverse=False)
        p_density_profile_sorted = [x for _, x, _ in sorted_zips_p]
        p_error_bar_sorted = [x for _, _, x in sorted_zips_p]
        sorted_zips_n = sorted(zip(z_val, negativeion_density_profile, n_error_bar),
                               key=lambda tup: tup[0], reverse=False)
        n_density_profile_sorted = [x for _, x, _ in sorted_zips_n]
        n_error_bar_sorted = [x for _, _, x in sorted_zips_n]
        z_val_sorted = [x for x, _, _ in sorted_zips_n]

        # 6. Concatenated lists -> string and write to file
        positive_strg = ' '.join(map(str, z_val_sorted + p_density_profile_sorted + p_error_bar_sorted))
        negative_strg = ' '.join(map(str, z_val_sorted + n_density_profile_sorted + n_error_bar_sorted))
        f_pos.write(positive_strg + "\n")  #z_val_strg + pos_den_strg + pos_err_strg + "\n")
        f_neg.write(negative_strg + "\n")  #z_val_strg + neg_den_strg + neg_err_strg + "\n")
        f_pos.close()
        f_neg.close()
        return












        # string_rho_p = ""
        # string_rho_n = ""
        # string_midp = ""
        # string_error_p = ""
        # string_error_n = ""
        # print("DEBUG:::::len(bins_sorted):", len(bins_sorted))
        # for b in range(0, len(bins_sorted)):
        #     # stringRow_p = str(bins_sorted[b].midpoint * utility.unitlength)+"\t"+str(positiveion_density_profile[b])+"\t"+str(p_error_bar[b])+"\n"
        #     string_rho_p += str(positiveion_density_profile[b])+" "
        #     string_midp += str(bins_sorted[b].midpoint * utility.unitlength)+" "
        #     string_error_p += str(p_error_bar[b])+" "
        #     # positiveDenistyMap[bins_sorted[b].midpoint * utility.unitlength] = stringRow_p
        #     string_rho_n += str(negativeion_density_profile[b]) + "\t"
        #     string_error_n += str(n_error_bar[b]) + "\t"
        #     # stringRow_n = str(bins_sorted[b].midpoint * utility.unitlength)+"\t"+str(negativeion_density_profile[b])+"\t"+str(n_error_bar[b])+"\n"
        #     # negativeDensityMap[bins_sorted[b].midpoint * utility.unitlength] = stringRow_n
        # f_pos.write(string_midp + string_rho_p + string_error_p+"\n")
        # f_neg.write(string_midp + string_rho_n + string_error_n+"\n")

        # for key in positiveDenistyMap.keys():
        #     f_pos.write(positiveDenistyMap[key])
        # for key in negativeDensityMap.keys():
        #     f_neg.write(negativeDensityMap[key])
