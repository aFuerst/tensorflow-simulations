import pandas as pd
import os, os.path, pickle, subprocess, time
import timeit

def run_simulation(totaltime, steps, log_freq, number_ljatom, xla, force_cpu, optimize):
    args = []
    args.append(os.path.expanduser("/extra/alfuerst/anaconda3/bin/python"))
    args.append(os.path.expanduser("~/tensorflow-simulations/lj-box/main.py"))
    if optimize:
        args.append("--opt")
    if force_cpu:
        args.append("--cpu")
    if xla:
        args.append("--xla")
        subprocess.run(["TF_XLA_FLAGS=--tf_xla_auto_jit=2"], shell=True)
    args.append("-p {0}".format(number_ljatom))
    args.append("-s {0}".format(steps))
    args.append("-t {0}".format(totaltime))
    args.append("-l {0}".format(log_freq))
    beg = time.time()
    sim = subprocess.run(args, capture_output=True)
    ret = time.time() - beg
    try:
        simul_time = float(sim.stdout)
    except:
        print(args)
        print(sim.stdout)
        print()
        raise
    subprocess.run(["TF_XLA_FLAGS=0"], shell=True)
    return ret, simul_time

save=[]
for num_atoms in range(100, 2000, 100):
    optimized = timeit.timeit(lambda: run_simulation(totaltime=10, steps=10000, log_freq=1000, number_ljatom=num_atoms, xla=True, force_cpu=False, optimize=True), number=10)
    no_optimizations = timeit.timeit(lambda: run_simulation(totaltime=10, steps=10000, log_freq=1000, number_ljatom=num_atoms, xla=True, force_cpu=False, optimize=False), number=10)
    save.append((num_atoms, optimized, no_optimizations))

print(save)