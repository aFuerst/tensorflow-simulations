import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import tensorflow as tf
import numpy as np
import math, sys

class Box:
    def __init__(self, x_len: float=10.0, y_len: float=10.0, z_len: float=10.0, df_type=np.float64):
        self._x = x_len
        self._y = y_len
        self._z = z_len
        self._atoms = None
        self._masses = None
        self._diameters = None
        self._df_type = df_type

    def fill(self, num_atoms, atom_diam):
        num_atoms_linear = int(math.ceil(pow(num_atoms,1.0/3.0)))
        self._atoms = np.zeros((0, 3), dtype=self._df_type)
        self._masses = np.zeros((0, 1), dtype=self._df_type)
        self._diameters = np.zeros((0, 1), dtype=self._df_type)
        a = 1
        for i in range(num_atoms_linear):
            for j in range(num_atoms_linear):
                for k in range(num_atoms_linear):
                    # breaking condition
                    if self._atoms.shape[0] >= num_atoms:
                        return self._atoms, self._atoms.shape[0], self._masses, self._diameters
                    else:
                        # start filling
                        x = (-self._x/2 + a/2.0) + i*a
                        y = (-self._y/2 + a/2.0) + j*a
                        z = (-self._z/2 + a/2.0) + k*a
                        #if z - (self._z/2.0 - a/2.0) > 0.0001 or y - (self._y/2.0 - a/2.0) > 0.0001 or x - (self._x/2.0 - a/2.0) > 0.0001:
                            #print("\n",z," ",self._z/2.0 - a/2.0)
                        if z >= self._z/2.0 - a/2.0 or y >= self._y/2.0 - a/2.0 or x >= self._x/2.0 - a/2.0:
                            continue
                        else:
                            new = np.asarray([[x,y,z]], dtype=self._df_type)
                            self._atoms = np.concatenate([self._atoms, new], axis=0)
                            self._masses = np.concatenate([self._masses, [[1]]], axis=0)
                            self._diameters = np.concatenate([self._diameters, [[atom_diam]]], axis=0)
        if self._atoms.shape[0] < num_atoms:
            print("WARNING: Unable to fit in {0} atoms, could only place {1}".format(num_atoms, self._atoms.shape[0]), file=sys.stderr)  #stop here
        # print("atoms coordinates are:",self._atoms[0:100])
        # print(self._atoms[100:200])
        # print(self._atoms[200:300])
        # print(self._atoms[300:344])
        return self._atoms, self._atoms.shape[0], self._masses, self._diameters

    def get_edges_as_tf(self):
        return tf.constant(np.array([self._x, self._y, self._z]), name="edges", dtype=self._df_type)

# if __name__ == "__main__":
#     size = 2.51939*2
#     b = Box(size,size,size)
#     atoms = b.fill(5, 1)
#     print(atoms.shape)
#     b = Box(size,size,size)
#     atoms = b.fill(500, 1)
#     print(atoms.shape)