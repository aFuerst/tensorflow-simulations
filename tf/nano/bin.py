import utility
import tensorflow as tf

def make_bins(box, bin_width, ion_diams):
    bins = []
    number_of_bins = int(box.lz / bin_width)
    bin_width = (box.lz / number_of_bins) #  To make discretization of bins symmetric, we recalculate the bin_width
    # Add two extra bins for contact point densities at both ends
    number_of_bins += 2
    for bin_num in range(number_of_bins):
        bins.append(Bin(bin_num, bin_width, box.lx, box.ly, box.lz))
    leftContact = -0.5 * box.lz + 0.5 * ion_diams[0] - 0.5 * bins[0].width
    rightContact = 0.5 * box.lz - 0.5 * ion_diams[0] - 0.5 * bins[0].width
    bins[len(bins) -1].lower = leftContact
    bins[len(bins) - 2].lower = rightContact
    bins[len(bins) - 1].higher = leftContact + bins[0].width
    bins[len(bins) - 2].higher = rightContact + bins[0].width
    bins[len(bins) - 1].midPoint = 0.5 * (bins[len(bins) - 1].lower + bins[len(bins) - 1].higher)
    bins[len(bins) - 2].midPoint = 0.5 * (bins[len(bins) - 2].lower + bins[len(bins) - 2].higher)
    return bins

@tf.function
def tf_get_ion_bin_density(box, ions, bins):
    r = ions[:, 2] # get z-axis value
    r = r + (0.5*box.lz)
    bin_nums = r / bins[0].width 
    bin_nums = tf.dtypes.cast(bin_nums, tf.int32)
    # bin_nums = ints[:, 2]
    return tf.sort(bin_nums) 

# converted from 'bin_ions'
def get_ion_bin_density(box, ions, bins):
    r = None
    bin_number = 0

    for bin in bins:
        bin.n = 0
    for i in range(len(ions)):
        # 0th bin is for left wall bin 0 starts at -0.5*box.lz respect to ion origin 
        r = 0.5 * box.lz + ion[i].posvec.z    # r should be positive, counting from half lz and that is the adjustment here bin 0 corresponds to left wall
        bin_number = int(r / bins[0].width)
        bins[bin_number].n = bins[bin_number].n + 1
    # This is to get contact point densities
    for i in range(len(ions)):
        # 0th bin is for left wall bin 0 starts at -0.5*box.lz respect to ion origin */
        r = 0.5 * box.lz + ion[i].posvec.z
        if (bins[len(bins) - 2].lower <= ion[i].posvec.z and ion[i].posvec.z < bins[len(bins) - 2].higher):
            bins[len(bins) - 2].n = bins[len(bins) - 2].n + 1
        elif (bins[len(bins) - 1].lower <= ion[i].posvec.z and ion[i].posvec.z < bins[len(bins) - 1].higher):
            bins[len(bins) - 1].n = bins[len(bins) - 1].n + 1

    density = []
    for bin_num in range(len(bins)):
        density.append(bins[bin_num].n / bins[bin_num].volume) # push_back is the culprit, array goes out of bound
    # volume now measured in inverse Molars, such that density is in M

class Bin:
    def __init__(self, bin_num: int, bin_width: float, lx: float, ly: float, lz: float):
        self.n = 0       # initialize number of ions in bin to be zero
        self.width = bin_width
        self.volume = (bin_width * lx * ly) * (utility.unitlength * utility.unitlength * utility.unitlength) * 0.6022
        self.lower = 0.5 * (-lz) + bin_num * bin_width
        self.higher = 0.5 * (-lz) + (bin_num + 1) * bin_width
        self.midPoint = 0.5 * (self.lower + self.higher)