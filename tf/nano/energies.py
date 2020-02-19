import tensorflow as tf
import numpy as np

import common, utility

def _unknown():
    """
    charged sheets method is used to compute Coulomb interactions; an Ewald version should be designed to compare and ensure that long-range effects are taken into account in either methods
    """
    pass

def _lj_energy():
    """
    Excluded volume interaction energy given by purely repulsive LJ
    ion-ion
    """
    pass

def _left_wall_lj_energy():
    """
    left wall
    ion interacting with left wall directly (self, closest)
    """
    pass

def _right_wall_lj_energy():
    """
    right wall
    ion interacting with right wall directly (self, closest)
    """
    pass

def _right_wall_columb_energy():
    """
    coulomb interaction ion-rightwall
    """
    pass

def _left_wall_columb_energy():
    """
    coulomb interaction ion-leftwall
    """
    pass

def energy_functional():
    pass