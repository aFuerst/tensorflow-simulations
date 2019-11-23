import main
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plot
import pandas as pd
import os, os.path

def save_plot(path):
    if os.path.isfile(path):
        os.remove(path)
    plot.savefig(path, bbox_inches="tight")

fig, ax = plot.subplots()
pts = []
for num_atoms in range(100, 2000, 100):
    times, total = main.run_simulation(totaltime=10, steps=10000, log_freq=1000, number_ljatom=num_atoms)
    df = pd.DataFrame.from_records(times, columns=["step", "time"])
    ax.plot(df["step"], df["time"], label=str(num_atoms))
    pts.append((num_atoms, total))

ax.legend()
save_plot("./t1.png")

fig, ax = plot.subplots()
pts = sorted(pts, key= lambda x: x[0])
ax.plot([x for x,y in pts],[y for x,y in pts], color="tab:green")
ax.set_xlabel("Num Atoms")
ax.set_ylabel("Time (s)")
ax.set_title("Simulation duration")
save_plot("./t2.png")

fig, ax = plot.subplots()
pts = []
for num_steps in range(10000, 100000, 10000):
    times, total = main.run_simulation(totaltime=num_steps//1000, steps=num_steps, log_freq=num_steps//10, number_ljatom=108)
    df = pd.DataFrame.from_records(times, columns=["step", "time"])
    ax.plot(df["step"], df["time"], label=str(num_steps))
    pts.append((num_steps, total))

ax.legend()
save_plot("./t1.png")

fig, ax = plot.subplots()
pts = sorted(pts, key= lambda x: x[0])
ax.plot([x for x,y in pts],[y for x,y in pts], color="tab:green")
ax.set_xlabel("Num Atoms")
ax.set_ylabel("Time (s)")
ax.set_title("Simulation duration")
save_plot("./t2.png")