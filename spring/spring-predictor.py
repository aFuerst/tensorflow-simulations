import numpy as np
import tensorflow as tf
import pandas as pd
import sys, math, time, argparse, os, os.path

class spring_gen(tf.keras.utils.Sequence):
        def __init__(self):
            self._data_headers = ["#", "position", "kinetic", "potential", "total", "energy", "velocity"]
            self._data_path = "/extra/alfuerst/sim/spring/"
            self._data_paths = list(map(lambda x: os.path.join(self._data_path, x), os.listdir(self._data_path)))
            self._xs = []
            self._ys = []
            for i, path in enumerate(self._data_paths):
                mass, k, a = self._get_params(path)
                df = pd.read_csv(path, names=self._data_headers, skiprows=1, sep="\s+")
                w = self._get_w(df, a, path)
                a_k = self._get_avg_k(df)
                self._xs.append(np.array([mass, k, a]))
                self._ys.append(np.array([w]))
            self._ys = np.asarray(self._ys)
            self._xs = np.asarray(self._xs)
            self._batch_size = 100

        def __len__(self):
            return len(self._data_paths)

        def _get_params(self, path):
            name = path.split("/")[-1]
            mass, k, a = name[0:-4].split("_")[0:3]
            return float(mass), float(k), float(a)

        def _get_w(self, df, a, path):
            spots = df.loc[df["position"]==a]
            first = 0
            seen = first
            pos = None
            for spot in spots.itertuples():
                if spot.Index == seen or spot.Index == seen+1:
                    continue
                else:
                    pos = spot
                    break
            try:
                return spots.loc[pos.Index]["#"]
            except:
                # spring never had a full cycle
                return -1

        def _get_avg_k(self, pd):
            return pd["kinetic"].mean()

        # def _get_w(self, path):
        #     pass

        def __getitem__(self, idx):
            idx = idx*self._batch_size
            return self._xs[idx:self._batch_size+idx], self._ys[idx:self._batch_size+idx]

relu = tf.keras.activations.relu
model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(3)),
    tf.keras.layers.Dense(5, relu),
    tf.keras.layers.Dense(5, relu),
    tf.keras.layers.Dense(5, relu),
    tf.keras.layers.Dense(5, relu),
    tf.keras.layers.Dense(1) # linear 
])
model.compile(tf.keras.optimizers.Adam(), loss='mae', metrics=['accuracy'])

g = spring_gen()
hist = model.fit_generator(g, epochs=50)
model.save("/home/alfuerst/tensorflow-simulations/spring/model.h5")

import matplotlib.pyplot as plot

plot.plot(hist.history['acc'])
plot.ylabel('acc')
plot.xlabel('Epoch')
plot.savefig("accuracy.png", bbox_inches="tight")
plot.clf()
plot.cla()
plot.plot(hist.history['loss'])
plot.ylabel('loss')
plot.xlabel('Epoch')
plot.savefig("loss.png", bbox_inches="tight")