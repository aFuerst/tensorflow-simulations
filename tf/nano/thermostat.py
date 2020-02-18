import tensorflow as tf
import numpy as np
import utility

def make_thremostats(chain_length_real, ions_count):
    Q = 1.0 # thremostat mass
    therms = {"Q":[],"T":[],"dof":[],"xi":[],"eta":[],"hold":[]}
    if (chain_length_real == 1):
        therms["Q"].append(0.0)
        therms["T"].append(utility.T)
        therms["dof"].append(3 * ions_count)
        therms["xi"].append(0.0)
        therms["eta"].append(0.0)
        therms["hold"].append(0.0)
        # .push_back((THERMOSTAT(0, T, 3 * ion.size(), 0.0, 0, 0)));
    else:
        therms["Q"].append(Q)
        therms["T"].append(utility.T)
        therms["dof"].append(3 * ions_count)
        therms["xi"].append(0.0)
        therms["eta"].append(0.0)
        therms["hold"].append(0.0)
        # real_bath.push_back((THERMOSTAT(Q, T, 3 * ion.size(), 0, 0, 0)));
        while (len(therms["Q"]) != chain_length_real - 1):
            therms["Q"].append(Q / (3 * ions_count))
            therms["T"].append(utility.T)
            therms["dof"].append(1.0)
            therms["xi"].append(0.0)
            therms["eta"].append(0.0)
            therms["hold"].append(0.0)
            # real_bath.push_back((THERMOSTAT(Q / (3 * ion.size()), T, 1, 0, 0, 0)));
        therms["Q"].append(Q)
        therms["T"].append(utility.T)
        therms["dof"].append(3 * ions_count)
        therms["xi"].append(0.0)
        therms["eta"].append(0.0)
        therms["hold"].append(0.0)
        # real_bath.push_back((THERMOSTAT(0, T, 3 * ion.size(), 0.0, 0, 0)));
        # final bath is dummy bath (dummy bath always has zero mass)
    for key in therms.keys():
        therms[key] = np.array(therms[key])
    return therms
  
# returns full therms dictionary with updated xi tensor
# TODO: IT will probaly be a tensor, not float
@tf.function
def update_xi(therms, IT: float, dt: float):
    with tf.name_scope("update_xi"):
        xi = therms["xi"] + 0.5 * dt * (1.0 / therms["Q"]) * (IT - therms["dof"] * utility.kB * utility.T)
        filt = therms["Q"] == 0
        therms["xi"] =  tf.compat.v1.where_v2(filt, xi, therms["xi"], name="therm_xi_where")
        return therms
    # if (Q == 0):
    #   return
    # xi = xi + 0.5 * dt * (1.0 / Q) * (IT - dof * utility.kB * utility.T) # not used. valid only for a solitary thermostat.

# returns full therms dictionary with updated eta tensor
@tf.function
def update_eta(therms, dt: float):
    with tf.name_scope("update_eta"):
        therms["eta"] = tf.compat.v1.where_v2(therms["Q"] == 0, therms["eta"] + 0.5 * dt * therms["xi"], therms["eta"], name="therm_eta_where")
        return therms
    # if (Q == 0):
    #   return
    # eta = eta + 0.5 * dt * xi

# calculate potential energy
@tf.function
def potential_energy(therms):
    with tf.name_scope("therms_potential_energy"):
        return therms["dof"] * utility.kB * utility.T * therms["eta"]
    # pe = dof * kB * T * eta # eta is zero for dummy making pe 0

# calculate kinetic energy
@tf.function
def kinetic_energytherms(therms):
    with tf.name_scope("therms_kinetic_energytherms"):
        return 0.5 * therms["Q"] * therms["xi"] * therms["xi"]
        # ke = 0.5 * Q * xi * xi # Q is zero for dummy making ke 0