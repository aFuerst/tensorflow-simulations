import main
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plot
import pandas as pd
import os, os.path, pickle

def save_plot(path):
    if os.path.isfile(path):
        os.remove(path)
    plot.savefig(path, bbox_inches="tight")

atoms_labels = ["gpu", "cpu", "xla+gpu", "xla+cpu"]
steps_labes = ["steps-gpu", "steps-cpu", "steps-xla+gpu", "steps-xla+cpu"]
saves = None
with open("./xla-timings.pckl", "r+b") as f:
    saves = pickle.load(f)

######### ploting num atoms #########

fig, ax = plot.subplots()

for label in atoms_labels:
    df = pd.DataFrame.from_records(saves[label], columns=["atoms", "time"])
    ax.plot(df["atoms"], df["time"], label=label)

open_mp = "/extra/alfuerst/sim/omp-out/atoms-tims"
pts = []
for direct in os.listdir(open_mp):
    atoms = int(direct)
    path = os.path.join(open_mp, direct, "times-{}".format(atoms))
    times = open(path).readlines()
    start = float(times[0])
    stop = float(times[1])
    pts.append((atoms, stop-start))
pts = sorted(pts, key=lambda x: x[0])
df = pd.DataFrame.from_records(pts, columns=["atoms", "time"])
ax.plot(df["atoms"], df["time"], label="OpenMP")

ax.legend()
ax.set_xlabel("Num Atoms")
ax.set_ylabel("Time (s)")
# ax.set_title("Simulation duration")
save_plot("./atoms.png")

######### end ploting num atoms #########

######### ploting num steps #########

fig, ax = plot.subplots()
for label in steps_labes:
    df = pd.DataFrame.from_records(saves[label], columns=["step", "time"])
    ax.plot(df["step"], df["time"], label=label)

open_mp = "/extra/alfuerst/sim/omp-out/steps-tims"
pts = []
for direct in os.listdir(open_mp):
    steps = int(direct)
    path = os.path.join(open_mp, direct, "times-{}".format(steps))
    times = open(path).readlines()
    start = float(times[0])
    stop = float(times[1])
    pts.append((steps, stop-start))
pts = sorted(pts, key=lambda x: x[0])
df = pd.DataFrame.from_records(pts, columns=["steps", "time"])
ax.plot(df["steps"], df["time"], label="OpenMP")

ax.legend()
ax.set_xlabel("Num Steps")
ax.set_ylabel("Time (s)")
# ax.set_title("Simulation duration")
save_plot("./steps.png")

######### end ploting num steps #########

atoms_labels = ["xla+gpu", "xla+cpu"]
saves = None
with open("./xla-timings.pckl", "r+b") as f:
    saves = pickle.load(f)

fig, ax = plot.subplots()

for label in atoms_labels:
    df = pd.DataFrame.from_records(saves[label], columns=["atoms", "time"])
    ax.plot(df["atoms"], df["time"], label=label)

open_mp = "/extra/alfuerst/sim/omp-out/atoms-tims"
pts = []
for direct in os.listdir(open_mp):
    atoms = int(direct)
    path = os.path.join(open_mp, direct, "times-{}".format(atoms))
    times = open(path).readlines()
    start = float(times[0])
    stop = float(times[1])
    pts.append((atoms, stop-start))
pts = sorted(pts, key=lambda x: x[0])
df = pd.DataFrame.from_records(pts, columns=["atoms", "time"])
ax.plot(df["atoms"], df["time"], label="OpenMP")

open_mp = "/home/alfuerst/tensorflow-simulations/lj-box/lin-cpp-out"
pts = []
for direct in os.listdir(open_mp):
    atoms = int(direct)
    path = os.path.join(open_mp, direct, "times-{}".format(atoms))
    times = open(path).readlines()
    start = float(times[0])
    stop = float(times[1])
    pts.append((atoms, stop-start))
pts = sorted(pts, key=lambda x: x[0])
df = pd.DataFrame.from_records(pts, columns=["atoms", "time"])

ax.legend()
ax.set_xlabel("Num Atoms")
ax.set_ylabel("Time (s)")
save_plot("./openmp-tf-tims.png")

ax.plot(df["atoms"], df["time"], label="CPP")

ax.legend()
ax.set_xlabel("Num Atoms")
ax.set_ylabel("Time (s)")
save_plot("./cpp-tf-tims.png")