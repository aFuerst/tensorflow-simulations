import tensorflow as tf
import numpy as np

import common, utility

def _left_wall_lj_force():
    """
    ion-box
    interaction with the left plane hard wall
    make a dummy particle with the same diameter as the ion and touching left of the left wall s. t. it is closest to the ion
    """
    pass

def _right_wall_lj_force():
    """
    interaction with the right plane hard wall
    make a dummy particle with the same diameter as the ion and touching right of the right wall s. t. it is closest to the ion
    """
    pass

def _particle_lj_force():
    """
    excluded volume interactions given by purely repulsive LJ
    ion-ion
    """
    pass

def _particle_electrostatic_force():
    """
    force on the particles (electrostatic)
    parallel calculation of forces (uniform case)
    """
    pass

def _electrostatic_right_wall_force():
    """
    ion interacting with discretized right wall
    electrostatic between ion and rightwall
    """
    pass

def _electrostatic_left_wall_force():
    """
    ion interacting with discretized left wall
    electrostatic between ion and left wall
    """
    pass

def for_md_calculate_force(sumil_box, thermostats, ion_dict, bins, charge_meshpoint: float, valency_counterion: int):
    unsigned int range = ion.size() / world.size() + 1.5
    unsigned int lowerBound = world.rank() * range
    unsigned int upperBound = (world.rank() + 1) * range - 1
    unsigned int extraElements = world.size() * range - ion.size()
    unsigned int sizFVec = upperBound - lowerBound + 1
    if (world.rank() == world.size() - 1) {
        upperBound = ion.size() - 1
        sizFVec = upperBound - lowerBound + 1 + extraElements
    }
    if (world.size() == 1) {
        lowerBound = 0
        upperBound = ion.size() - 1
    }
    pass