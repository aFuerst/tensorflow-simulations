import numpy as np
import tensorflow as tf
import pandas as pd
import sys, math, time, argparse, os, os.path

model = tf.keras.models.load_model("/home/alfuerst/tensorflow-simulations/spring/model.h5")

def get_w(df, a):
    spots = df.loc[df["position"]==a]
    first = 0
    seen = first
    print(spots, a)
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

def get_avg_k(pd):
    return pd["kinetic"].mean()

mass = 2
k = 1
a = -1

beg = time.time()
data_headers = ["#", "position", "kinetic", "potential", "total", "energy", "velocity"]
df = pd.read_csv("/extra/alfuerst/sim/spring/2_1_-1.txt", names=data_headers, skiprows=1, sep="\s+")
print(df)
w = get_w(df, a)
a_k = get_avg_k(df)
print("actual", w, a_k)
print("sim time", time.time() - beg)

p = model.predict(np.array([[mass, k, a]]))
beg = time.time()
p = model.predict(np.array([[mass, k, a]]))
print("predicted", p)
print("inf time", time.time() - beg)
