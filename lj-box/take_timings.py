# import main
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plot
import pandas as pd
import os, os.path, pickle, subprocess, time

def save_plot(path):
    if os.path.isfile(path):
        os.remove(path)
    plot.savefig(path, bbox_inches="tight")

fig, ax = plot.subplots()
pts = []
save = {}

def run_simulation(totaltime, steps, log_freq, number_ljatom, xla, force_cpu):
    args = []
    args.append(os.path.expanduser("~/anaconda3/bin/python"))
    args.append(os.path.expanduser("~/tensorflow-simulations/lj-box/main.py"))
    if force_cpu:
        args.append("--cpu")
    if xla:
        args.append("--xla")
        subprocess.run(["TF_XLA_FLAGS=--tf_xla_auto_jit=2"], shell=True)
    args.append("-p {0}".format(number_ljatom))
    args.append("-s {0}".format(steps))
    args.append("-t {0}".format(totaltime))
    args.append("-l {0}".format(log_freq))
    # print(args)
    beg = time.time()
    r = subprocess.run(args, capture_output=True)
    ret = time.time() - beg
    # print(r.stdout)
    print(float(r.stdout)) # main.py prints how long the computation step took
    subprocess.run(["TF_XLA_FLAGS=0"], shell=True)
    return ret

############### Atoms timing ###############

save["xla+gpu"] = []
for num_atoms in range(100, 2000, 100):
    total = run_simulation(totaltime=10, steps=10000, log_freq=1000, number_ljatom=num_atoms, xla=True, force_cpu=False)
    save["xla+gpu"].append((num_atoms, total))

save["gpu"] = []
for num_atoms in range(100, 2000, 100):
    total = run_simulation(totaltime=10, steps=10000, log_freq=1000, number_ljatom=num_atoms, xla=False, force_cpu=False)
    save["gpu"].append((num_atoms, total))

save["cpu"] = []
for num_atoms in range(100, 2000, 100):
    total = run_simulation(totaltime=10, steps=10000, log_freq=1000, number_ljatom=num_atoms, xla=False, force_cpu=True)
    save["cpu"].append((num_atoms, total))

save["xla+cpu"] = []
for num_atoms in range(100, 2000, 100):
    total = run_simulation(totaltime=10, steps=10000, log_freq=1000, number_ljatom=num_atoms, xla=True, force_cpu=True)
    save["xla+cpu"].append((num_atoms, total))

############### Steps timing ###############

save["steps-xla+cpu"] = []
for num_steps in range(10000, 210000, 10000):
    total = run_simulation(totaltime=10, steps=num_steps, log_freq=1000, xla=True, force_cpu=True, number_ljatom=108)
    save["steps-xla+cpu"].append((num_steps, total))

save["steps-xla+gpu"] = []
for num_steps in range(10000, 210000, 10000):
    total = run_simulation(totaltime=10, steps=num_steps, log_freq=1000, xla=True, force_cpu=False, number_ljatom=108)
    save["steps-xla+gpu"].append((num_steps, total))

save["steps-cpu"] = []
for num_steps in range(10000, 210000, 10000):
    total = run_simulation(totaltime=10, steps=num_steps, log_freq=1000, xla=False, force_cpu=True, number_ljatom=108)
    save["steps-cpu"].append((num_steps, total))

save["steps-gpu"] = []
for num_steps in range(10000, 210000, 10000):
    total = run_simulation(totaltime=10, steps=num_steps, log_freq=1000, xla=False, force_cpu=False, number_ljatom=108)
    save["steps-gpu"].append((num_steps, total))

with open("./xla-timings.pckl", "w+b") as f:
    pickle.dump(save, f)