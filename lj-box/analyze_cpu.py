import os
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plot

def save_plot(path):
    if os.path.isfile(path):
        os.remove(path)
    plot.savefig(path, bbox_inches="tight")

folder1 = "/home/alfuerst/tensorflow-simulations/lj-box/cpu_stats_108"
folder2 = "/home/alfuerst/tensorflow-simulations/lj-box/cpu_stats_2000"

headers = ["timestamp", "cpu"]

def get_thread_cnt(fname):
    p = fname.split(".")[0]
    p = p.split("_")[1]
    return int(p)
fig, ax = plot.subplots()

pts = []
for file in os.listdir(folder1):
    cnt = get_thread_cnt(file)
    path = os.path.join(folder1, file)
    df = pd.read_csv(path, names=headers, sep=' ')
    pts.append((cnt, df["cpu"].mean()))
    
pts = sorted(pts, key=lambda x: x[0])

df = pd.DataFrame.from_records(pts, columns=["threads", "cpu"])
ax.plot(df["threads"], df["cpu"], label="108 Atoms")

pts = []
for file in os.listdir(folder2):
    cnt = get_thread_cnt(file)
    path = os.path.join(folder2, file)
    df = pd.read_csv(path, names=headers, sep=' ')
    pts.append((cnt, df["cpu"].mean()))
    
pts = sorted(pts, key=lambda x: x[0])

df = pd.DataFrame.from_records(pts, columns=["threads", "cpu"])
ax.plot(df["threads"], df["cpu"], label="2000 Atoms")

ax.plot([i for i in range(5, 110, 5)], [min(i*100, 4800) for i in range(5, 110, 5)], label="Ideal")

ax.legend()
ax.set_xlabel("Allowed Threads")
ax.set_ylabel("Average CPU Usage")
save_plot("cpu.png")