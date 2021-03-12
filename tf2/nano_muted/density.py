import tensorflow as tf
import numpy as np

import utility, bin, common

def compute_density_profiles():
    # TODO: This
    # for density profile
    mean_positiveion_density = np.zeroes((len(bins),))            # average density profile
    mean_negativeion_density = np.zeroes((len(bins),))            # average density profile
    mean_sq_positiveion_density = np.zeroes((len(bins),))            # average of square of density
    mean_sq_negativeion_density = np.zeroes((len(bins),))            # average of square of density
