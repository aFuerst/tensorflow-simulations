from jax.config import config
config.update("jax_enable_x64", True)
import argparse
from jax import random, jit, lax
from jax_md import energy, quantity, space, simulate
import numpy as onp

import jax.numpy as np
import time, os

parser = argparse.ArgumentParser()
parser.add_argument('-p', "--parts", action="store", default=108, type=int)
parser.add_argument('-s', "--steps", action="store", default=10000, type=int)
parser.add_argument('-t', "--time", action="store", default=10, type=int)
parser.add_argument('-l', "--log", action="store", default=1000, type=int)
parser.add_argument('-d', "--dense", action="store", default=0.8442, type=float)
args = parser.parse_args()

edge_length = pow(args.parts/args.dense,1.0/3.0)
# edge_length*=2
spatial_dimension = 3
box_size = onp.asarray([edge_length]*spatial_dimension)
displacement_fn, shift_fn = space.periodic(box_size)
key = random.PRNGKey(0)
R = random.uniform(key, (args.parts, spatial_dimension), minval=0.0, maxval=box_size[0], dtype=np.float64)
# print(R)
energy_fn = energy.lennard_jones_pair(displacement_fn)
print('E = {}'.format(energy_fn(R)))
force_fn = quantity.force(energy_fn)
print('Total Squared Force = {}'.format(np.sum(force_fn(R) ** 2)))
init, apply = simulate.nve(energy_fn, shift_fn, args.time/args.steps)
apply = jit(apply)
state = init(key, R, velocity_scale=0.0)

PE = []
KE = []

print_every = args.log
old_time = time.perf_counter()
print('Step\tKE\tPE\tTotal Energy\ttime/step')
print('----------------------------------------')
for i in range(args.steps//print_every):
    state = lax.fori_loop(0, print_every, lambda _, state: apply(state), state)
    
    PE += [energy_fn(state.position)]
    KE += [quantity.kinetic_energy(state.velocity)]
    new_time = time.perf_counter()
    print('{}\t{:.2f}\t{:.2f}\t{:.3f}\t{:.2f}'.format(
        i*print_every, KE[-1], PE[-1], KE[-1] + PE[-1], 
        (new_time - old_time) / print_every / 10.0))
    old_time = new_time

PE = np.array(PE)
KE = np.array(KE)
R = state.position

import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
  
sns.set_style(style='white')

def format_plot(x, y):
  plt.xlabel(x, fontsize=20)
  plt.ylabel(y, fontsize=20)
  
def finalize_plot(shape=(1, 1)):
  plt.gcf().set_size_inches(
    shape[0] * 1.5 * plt.gcf().get_size_inches()[1], 
    shape[1] * 1.5 * plt.gcf().get_size_inches()[1])
  plt.tight_layout()
  os.remove("./jax-eng.png")
  plt.savefig("./jax-eng.png", bbox_inches="tight")
  
print(len(PE))
t = onp.arange(0, args.steps, print_every) * args.time/args.steps
print(t.shape)
plt.plot(t, PE, label='PE', linewidth=3)
plt.plot(t, KE, label='KE', linewidth=3)
plt.plot(t, PE + KE, label='Total Energy', linewidth=2)
plt.legend()
format_plot('t', '')
finalize_plot()