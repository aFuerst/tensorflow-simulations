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
for num_atoms in range(100, 1000, 100):
    times, total = main.run_simulation(totaltime=10, steps=10000, log_freq=1000, number_ljatom=num_atoms)
    df = pd.DataFrame.from_records(times, columns=["step", "time"])
    ax.plot(df["step"], df["time"])

ax.legend()
save_plot("./t1.png")