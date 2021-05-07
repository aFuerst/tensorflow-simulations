import os

import matplotlib as mpl
import pandas as pd

mpl.use('Agg')
import matplotlib.pyplot as plot

def save_plot(path):
    if os.path.isfile(path):
        os.remove(path)
    plot.savefig(path, bbox_inches="tight")

folder1 = "/home/alfuerst/tensorflow-simulations/lj-box/cpu_stats_108"
folder2 = "/home/alfuerst/tensorflow-simulations/lj-box/cpu_stats_2000"
folder3 = "/home/alfuerst/tensorflow-simulations/lj-box/cpu_stats_5000"

headers = ["timestamp", "cpu"]

def get_thread_cnt(fname):
    p = fname.split(".")[0]
    p = p.split("_")[1]
    return int(p)
fig, ax = plot.subplots()

def get_pts(folder):
    pts = []
    for file in os.listdir(folder):
        cnt = get_thread_cnt(file)
        path = os.path.join(folder, file)
        df = pd.read_csv(path, names=headers, sep=' ')
        pts.append((cnt, df["cpu"].mean()))
        
    pts = sorted(pts, key=lambda x: x[0])
    df = pd.DataFrame.from_records(pts, columns=["threads", "cpu"])
    return df

df = get_pts(folder1)
ax.plot(df["threads"], df["cpu"], label="108 Atoms")
df = get_pts(folder2)
ax.plot(df["threads"], df["cpu"], label="2000 Atoms")
df = get_pts(folder3)
ax.plot(df["threads"], df["cpu"], label="5000 Atoms")

ax.plot([i for i in range(1, 101)], [min(i*100, 4800) for i in range(1, 101)], label="Ideal")

ax.legend()
ax.set_xlabel("Allowed Threads")
ax.set_ylabel("Average CPU Usage")
ax.axvline(x=48, color="red") # , label="# CPUs on Machine"
ax.text(49, 1300, "# Cores on Machine", fontsize=8, rotation=90)
ax.set_yticks([100,1000,2000,3000,4000, 4800])
ax.set_ylim([1, 4900])
ax.set_xticks([1,20, 40, 60, 80, 100])
ax.set_xlim([1, 100])
save_plot("cpu.png")