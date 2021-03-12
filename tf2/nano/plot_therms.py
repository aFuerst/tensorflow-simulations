import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plot
import os, os.path, itertools
import pandas as pd

marker = itertools.cycle(('o', '+', '.', 'x', '*'))
markerevery = itertools.cycle((6,2,3,4,5))

def save_plot(path):
    if os.path.isfile(path):
        os.remove(path)
    plot.savefig(path, bbox_inches="tight")

def get_step(file):
    return int(file.split("-")[0])

therms = {0:[], 1:[], 2:[], 3:[], 4:[]}
# therms_eta = {0:[], 1:[], 2:[], 3:[], 4:[]}
# therms_xi = {0:[], 1:[]}
# therms_eta = {0:[], 1:[]}
therms_xi = {}
therms_eta = {}
steps = []
folder = "./output_850_2"
for file in os.listdir(folder):
    if "thermostats" in file:
        step = get_step(file)
        steps.append(step)
        file = os.path.join(folder, file)
        with open(file) as f:
            for line in f.readlines():
                therm = int(line[0])
                if therm not in therms_xi:
                    therms_xi[therm] = []
                    therms_eta[therm] = []
                xi, eta = line[2:].split(", ")
                print(xi, xi.split(":")[1], float(xi.split(":")[1]), eta, eta.split(":")[1], float(eta.split(":")[1]))
                therms_eta[therm].append((float(eta.split(":")[1]), step))
                therms_xi[therm].append((float(xi.split(":")[1]), step))
                # therms_eta[therm].append()
# steps=sorted(steps)
fig, ax = plot.subplots()

for key, value in therms_xi.items():
    pts = sorted(value, key = lambda x: x[1])
    df = pd.DataFrame.from_records(pts, columns=["xi", "steps"])

    ax.plot(df["steps"], df["xi"], label="xi_"+str(key),
                marker=marker.__next__(), markevery=markerevery.__next__())

ax.legend()
ax.set_xlabel("Sim Step")
ax.set_ylabel("Xi")
save_plot("./figs/xi.png")

fig, ax = plot.subplots()

for key, value in therms_eta.items():
    pts = sorted(value, key = lambda x: x[1])
    df = pd.DataFrame.from_records(pts, columns=["eta", "steps"])
    print(df["eta"].describe())
    ax.plot(df["steps"], df["eta"], label="eta_"+str(key), marker=marker.__next__(), markevery=markerevery.__next__())

ax.legend()
ax.set_xlabel("Sim Step")
ax.set_ylabel("eta")
save_plot("./figs/eta.png")
