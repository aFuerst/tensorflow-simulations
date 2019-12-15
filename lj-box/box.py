import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import tensorflow as tf
import numpy as np
import math, sys

class Box:
    def __init__(self, x_len: float=10.0, y_len: float=10.0, z_len: float=10.0):
        self._x = x_len
        self._y = y_len
        self._z = z_len

    def fill(self, num_atoms, atom_diam):
        num_atoms_linear = int(math.ceil(pow(num_atoms,1.0/3.0)))
        atoms = np.zeros((0, 3), dtype=np.float64)
        a = 1
        for i in range(num_atoms_linear):
            for j in range(num_atoms_linear):
                for k in range(num_atoms_linear):
                    if atoms.shape[0] >= num_atoms:
                        return atoms, atoms.shape[0]
                    else:
                        x = (-self._x/2 + a/2.0) + i*a
                        y = (-self._y/2 + a/2.0) + j*a
                        z = (-self._z/2 + a/2.0) + k*a
                        if z >= self._z/2.0 - a/2.0 or y >= self._y/2.0 - a/2.0 or x >= self._x/2.0 - a/2.0:
                            continue
                        else:
                            new = np.asarray([[x,y,z]], dtype=np.float64)
                            atoms = np.concatenate([atoms, new], axis=0)
        if atoms.shape[0] < num_atoms:
            print("WARNING: Unable to fit in {0} atoms, could only place {1}".format(num_atoms, atoms.shape[0]), file=sys.stderr)
        return atoms, atoms.shape[0]

    def get_edges_as_tf(self):
        return tf.constant(np.array([self._x, self._y, self._z]), name="edges")

if __name__ == "__main__":
    size = 2.51939*2
    b = Box(size,size,size)
    atoms = b.fill(5, 1)
    print(atoms.shape)
    b = Box(size,size,size)
    atoms = b.fill(500, 1)
    print(atoms.shape)