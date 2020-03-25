import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plot
import os, os.path

def save_plot(path):
    if os.path.isfile(path):
        os.remove(path)
    plot.savefig(path, bbox_inches="tight")

def get_step(file):
    return int(file.split("-")[0])

therms_xi = {0:[], 1:[], 2:[], 3:[], 4:[]}
therms_eta = {0:[], 1:[], 2:[], 3:[], 4:[]}
steps = []
folder = "./output" 
for file in os.listdir(folder):
    if "thermostats" in file:
        step = get_step(file)
        steps.append(step)
        file = os.path.join(folder, file)
        with open(file) as f:
            for line in f.readlines():
                therm = int(line[0])
                xi, eta = line[2:].split(", ")
                therms_xi[therm].append(float(xi.split(":")[1]))
                therms_eta[therm].append(float(eta.split(":")[1]))
steps=sorted(steps)
fig, ax = plot.subplots()

for key, value in therms_xi.items():
    ax.plot(steps, value, label="xi_"+str(key))

ax.legend()
ax.set_xlabel("Sim Step")
ax.set_ylabel("Xi")
save_plot("./figs/xi.png")

fig, ax = plot.subplots()

for key, value in therms_eta.items():
    ax.plot(steps, value, label="eta_"+str(key))

ax.legend()
ax.set_xlabel("Sim Step")
ax.set_ylabel("eta")
save_plot("./figs/eta.png")