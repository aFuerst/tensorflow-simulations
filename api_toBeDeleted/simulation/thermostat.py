import tensorflow as tf
import numpy as np
from util import utility, common

_therm_constants = []


def to_string(therm_dict):
    return "xi:{}, eta:{}".format(therm_dict["xi"], therm_dict["eta"])


def create_thermostat(i, Q: float, T: float, dof: float, xi: float, eta: float, hold: float):
    _therm_constants.append({"Q": Q, "T": T, "dof": dof, "hold": hold})
    return {"xi": xi, "eta": eta}


def get_placeholders(therms):
    ret = []
    ret_copy = []
    for value in therms:
        place, copy = common.make_tf_placeholder_of_dict(value)
        ret.append(place)
        ret_copy.append(copy)
    return ret, ret_copy


def therms_to_feed_dict(therms, therms_place):
    feed = {}
    for therm_key, therm_value in enumerate(therms):
        for key, value in therm_value.items():
            feed[therms_place[therm_key][key]] = value
    return feed


def make_thermostats(chain_length_real, ions_count, Q):
    # Q = 1.0  # thremostat mass
    therms = []
    i = 0
    if (chain_length_real == 1):
        therms.append(create_thermostat(i=i, Q=0.0, T=utility.T, dof=3 * ions_count, xi=0.0, eta=0.0, hold=0.0))
    else:
        therms.append(create_thermostat(i=i, Q=Q, T=utility.T, dof=3 * ions_count, xi=0.0, eta=0.0, hold=0.0))
        i += 1
        while (len(therms) != chain_length_real - 1):
            therms.append(
                create_thermostat(i=i, Q=Q / (3 * ions_count), T=utility.T, dof=1.0, xi=0.0, eta=0.0, hold=0.0))
            i += 1
        # final therms is dummy therms (dummy therms always has zero mass)
        therms.append(create_thermostat(i=i, Q=0.0, T=utility.T, dof=3 * ions_count, xi=0.0, eta=0.0, hold=0.0))
    return therms


def update_xi_at(therms, j, dt, ke):
    # out_ke = tf.Print(ke,[ke],"ke")
    # print("j:",j," dt:",dt," ke:",ke)

    if _therm_constants[j]["Q"] == 0:
        # print("\n Q 0 for j=",j)
        return therms

    if (j != 0):
        therms[j]["xi"] = therms[j]["xi"] * tf.math.exp(-0.5 * dt * therms[j + 1]["xi"]) + 0.5 * dt * (
                1.0 / _therm_constants[j]["Q"]) * (
                                  _therm_constants[j - 1]["Q"] * therms[j - 1]["xi"] * therms[j - 1]["xi"] -
                                  _therm_constants[j]["dof"] * utility.kB * _therm_constants[j]["T"]) * tf.math.exp(
            -0.25 * dt * therms[j + 1]["xi"])
    else:
        therms[j]["xi"] = therms[j]["xi"] * tf.math.exp(-0.5 * dt * therms[j + 1]["xi"]) + (
                0.5 * dt * (1.0 / _therm_constants[j]["Q"]) * (
                2 * ke - _therm_constants[j]["dof"] * utility.kB * _therm_constants[j]["T"]) * tf.math.exp(
            -0.25 * dt * therms[j + 1]["xi"]))
        # (2 * ke - _therm_constants[j]["dof"] * utility.kB * _therm_constants[j]["T"])
    return therms


def reverse_update_xi(therms, dt: float, ke):
    with tf.name_scope("reverse_update_xi"):
        for j in range(len(therms) - 1, -1, -1):
            therms = update_xi_at(therms, j, dt, ke)
        return therms


def forward_update_xi(therms, dt: float, ke):
    with tf.name_scope("forward_update_xi"):
        for j in range(0, len(therms)):
            therms = update_xi_at(therms, j, dt, ke)
        return therms


def update_xi(therms, dt, ke, rev):
    if rev:
        therms = reverse_update_xi(therms, dt, ke)
    else:
        therms = forward_update_xi(therms, dt, ke)
    return therms


# returns full therms dictionary with updated eta tensor
def update_eta(therms, dt: float):
    with tf.name_scope("update_eta"):
        for j in range(0, len(therms)):
            if _therm_constants[j]["Q"] == 0:
                continue
            therms[j]["eta"] = therms[j]["eta"] + (0.5 * dt * therms[j]["xi"])
        return therms


def calc_exp_factor(therms, dt):
    with tf.name_scope("calc_exp_factor"):
        return tf.math.exp((-0.5) * dt * therms[0]["xi"])


def bath_potential_energy(therms):
    with tf.name_scope("bath_potential_energy"):
        pe_sum = 0.0
        for j in range(0, len(therms)):
            pe_sum = pe_sum + (_therm_constants[j]["dof"] * utility.kB * _therm_constants[j]["T"] * therms[j]["eta"])
            # pe.append(_therm_constants[j]["dof"] * utility.kB * _therm_constants[j]["T"] * therms[j]["eta"])
    return pe_sum


def bath_kinetic_energy(therms):
    with tf.name_scope("bath_kinetic_energy"):
        ke_sum = 0.0
        for j in range(0, len(therms)):
            # print("FOR j:",j," _therm_constants[j][Q]:", _therm_constants[j]["Q"], " T:", _therm_constants[j]["T"], " _therm_constants[j][dof]:", _therm_constants[j]["dof"]," therms[j][xi]",therms[j]["xi"]," therms[j][eta]:", therms[j]["eta"])
            # xi_sq = np.power(therms[j]["xi"], 2)
            ke_sum = ke_sum + (0.5 * _therm_constants[j]["Q"] * therms[j]["xi"] * therms[j]["xi"])
            # ke.append(0.5 * _therm_constants[j]["Q"] * therms[j]["xi"] * therms[j]["xi"])
    return ke_sum

