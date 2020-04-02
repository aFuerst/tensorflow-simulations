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

steps = []
ke_list = []
folder = "./output"
file = os.path.join(folder, "kinetic_energy")
with open(file) as f:
    for line in f.readlines():
        i, ke = line.split(":")
        steps.append(int(i))
        ke_list.append(float(ke))
fig, ax = plot.subplots()

ax.plot(steps, ke_list, label="Kinetic Energy", marker='*')

ax.legend()
ax.set_xlabel("Sim Step")
ax.set_ylabel("KE")
save_plot("./figs/ke.png")

steps = []
expfac_real_list = []
folder = "./output"
file = os.path.join(folder, "expfac_real")
with open(file) as f:
    for line in f.readlines():
        i, ke = line.split(":")
        steps.append(int(i))
        expfac_real_list.append(float(ke))
fig, ax = plot.subplots()

ax.plot(steps, expfac_real_list, label="expfac_real", marker='*')

ax.legend()
ax.set_xlabel("Sim Step")
ax.set_ylabel("expfac_real")
save_plot("./figs/expfac_real.png")

steps = []
expfac_real_list = []
folder = "./output"
file = os.path.join(folder, "temp")
with open(file) as f:
    for line in f.readlines():
        i, ke = line.split(":")
        steps.append(int(i))
        expfac_real_list.append(float(ke))
fig, ax = plot.subplots()

ax.plot(steps, expfac_real_list, label="temp", marker='*')

ax.legend()
ax.set_xlabel("Sim Step")
ax.set_ylabel("temp")
save_plot("./figs/temp.png")
