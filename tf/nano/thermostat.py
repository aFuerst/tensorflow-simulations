import tensorflow as tf
import numpy as np
import utility, common

def create_thermostat(i, Q:float, T:float, dof:float, xi:float, eta:float, hold:float):
    return {"Q":Q, "T":T, "dof":dof, "xi":xi, "eta":eta, "hold":hold}

def get_placeholders(therms):
    ret = {}
    ret_names = {}
    for key, value in therms.items():
        place, name = common.make_tf_placeholder_of_dict(value)
        ret[key] = place
        ret_names[key] = name
    return ret, ret_names

def therms_to_feed_dict(therms, therms_place):
    feed = {}
    for therm_key, therm_value in therms.items():
        for key, value in therm_value.items():
            feed[therms_place[therm_key][key]] = value
    return feed

def make_thremostats(chain_length_real, ions_count):
    Q = 1.0 # thremostat mass
    therms = {}
    i=0
    if (chain_length_real == 1):
        therms[i] = create_thermostat(i, 0.0, utility.T, 3* ions_count, 0.0, 0.0, 0.0)
    else:
        therms[i] = create_thermostat(i, Q, utility.T, 3* ions_count, 0.0, 0.0, 0.0)
        i += 1
        while (len(therms) != chain_length_real - 1):
            therms[i] = create_thermostat(i, Q / (3 * ions_count), utility.T, 1.0, 0.0, 0.0, 0.0)
            i += 1
        # final therms is dummy therms (dummy therms always has zero mass)
        therms[i] = create_thermostat(i, 0, utility.T, 3*ions_count, 0.0, 0.0, 0.0)
    return therms
  
# returns full therms dictionary with updated xi tensor
def update_chain_xi(therms, dt: float, ke):
    with tf.name_scope("update_chain_xi"):
        for j in range(len(therms)-1):
            if (j == len(therms)-1):
                raise Exception("can't change last j")
            if (j != 0):
                therms[j]["xi"] = therms[j]["xi"] * tf.math.exp(-0.5 * dt * therms[j + 1]["xi"]) + 0.5 * dt * (1.0 / therms[j]["Q"]) *\
                                                                    (therms[j - 1]["Q"] * therms[j - 1]["xi"] * therms[j - 1]["xi"] -\
                                                                     therms[j]["dof"] * utility.kB * therms[j]["T"]) *\
                                                                    tf.math.exp(-0.25 * dt * therms[j + 1]["xi"])
            else:
                therms[j]["xi"] = therms[j]["xi"] * tf.math.exp(-0.5 * dt * therms[j + 1]["xi"]) +\
                            0.5 * dt * (1.0 / therms[j]["Q"]) * (2 * ke - therms[j]["dof"] * utility.kB * therms[j]["T"]) *\
                            tf.math.exp(-0.25 * dt * therms[j + 1]["xi"])
        return therms

# returns full therms dictionary with updated eta tensor
def update_eta(therms, dt: float):
    with tf.name_scope("update_eta"):
        for j in range(len(therms)-1):
            therms[j]["eta"] = therms[j]["eta"] + 0.5 * dt * therms[j]["xi"]
        return therms

def calc_exp_factor(therms, dt):
    return tf.math.exp(-0.5 * dt * therms[0]["xi"])

if __name__ == "__main__":
    from tensorflow_manip import silence
    silence()
    r = make_thremostats(1, 5)
    print(len(r),r)
    x = update_chain_xi(r, 1.0, 1.0)
    print("\n result", x[0]["xi"])
    print()
    r = make_thremostats(5, 5)
    print(len(r),r)
    r_place = get_placeholders(r)

    print("\n\n", r_place)
    feed = therms_to_feed_dict(r, r_place)
    print(feed, "\n\n")
    x = update_chain_xi(r, 1.0, 1.0)
    print("\n result", x[0]["xi"])
    print(feed)
    sess = tf.compat.v1.Session()
    sess.as_default()
    print(sess.run(x[0]["xi"], feed_dict=feed))