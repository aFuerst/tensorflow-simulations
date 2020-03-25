import tensorflow as tf
import numpy as np
import utility, common

def to_string(therm_dict):
    return "xi:{}, eta:{}".format(therm_dict["xi"],therm_dict["eta"])

def create_thermostat(i, Q: float, T: float, dof: float, xi: float, eta: float, hold: float):
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
    Q = 1.0  # thremostat mass
    therms = {}
    i=0
    if (chain_length_real == 1):
        therms[i] = create_thermostat(i=i, Q=0.0, T=utility.T, dof=3* ions_count, xi=0.0, eta=0.0, hold=0.0)
    else:
        therms[i] = create_thermostat(i=i, Q=Q, T=utility.T, dof=3 * ions_count, xi=0.0, eta=0.0, hold=0.0)
        i += 1
        while (len(therms) != chain_length_real - 1):
            therms[i] = create_thermostat(i=i, Q=Q / (3 * ions_count), T=utility.T, dof=1.0, xi=0.0, eta=0.0, hold=0.0)
            i += 1
        # final therms is dummy therms (dummy therms always has zero mass)
    therms[i] = create_thermostat(i=i, Q=0.0, T=utility.T, dof=3*ions_count, xi=0.0, eta=0.0, hold=0.0)
    return therms

def update_xi_at(therms, j, dt, ke):
    if (j >= len(therms)-1):  # len(therms)-1 has Q==0 which is skipped
        raise Exception("can't change last j, {}".format(j))
    print("therm", j)
    if (j != 0):
        therms[j]["xi"] = therms[j]["xi"] * tf.math.exp(-0.5 * dt * therms[j + 1]["xi"]) + 0.5 * dt * (1.0 / therms[j]["Q"]) *\
                                                (therms[j - 1]["Q"] * therms[j - 1]["xi"] * therms[j - 1]["xi"] -
                                                    therms[j]["dof"] * utility.kB * therms[j]["T"]) *\
                                                tf.math.exp(-0.25 * dt * therms[j + 1]["xi"])
    else:
        therms[j]["xi"] = therms[j]["xi"] * tf.math.exp(-0.5 * dt * therms[j + 1]["xi"]) +\
                    (0.5 * dt * (1.0 / therms[j]["Q"]) * (2 * ke - therms[j]["dof"] * utility.kB * therms[j]["T"]) *\
                    tf.math.exp(-0.25 * dt * therms[j + 1]["xi"]))
    return therms

def reverse_update_xi(therms, dt: float, ke):
    with tf.name_scope("reverse_update_xi"):
        for j in range(len(therms)-2, -1, -1):
            therms = update_xi_at(therms, j, dt, ke)
        return therms
    
def forward_update_xi(therms, dt: float, ke):
    with tf.name_scope("forward_update_xi"):
        for j in range(0, len(therms)-1):
            therms = update_xi_at(therms, j, dt, ke)
        return therms        

# returns full therms dictionary with updated xi tensor
def update_chain_xi(therms, dt: float, ke, start: int, end: int):
    step = -1 if start > end else 1
    print("step", step)
    if len(therms) != 5:
        raise Exception("invalid number of thermostats")
    with tf.name_scope("update_chain_xi"):
        for j in range(start, end, step):  
            print("therm_xi", j)
            if (j >= len(therms)-1): # len(therms)-1 has Q==0 which is skipped
                raise Exception("can't change last j")
            if (j != 0):
                therms[j]["xi"] = therms[j]["xi"] * tf.math.exp(-0.5 * dt * therms[j + 1]["xi"]) + 0.5 * dt * (1.0 / therms[j]["Q"]) *\
                                                                    (therms[j - 1]["Q"] * therms[j - 1]["xi"] * therms[j - 1]["xi"] -\
                                                                     therms[j]["dof"] * utility.kB * therms[j]["T"]) *\
                                                                    tf.math.exp(-0.25 * dt * therms[j + 1]["xi"])
                #TODO!: REMOVE
                therms[j]["xi"] = tf.compat.v1.debugging.check_numerics(
                    therms[j]["xi"], message="therm_xi_not_zero")
                print(therms[j]["xi"])
            else:
                therms[j]["xi"] = therms[j]["xi"] * tf.math.exp(-0.5 * dt * therms[j + 1]["xi"]) +\
                            (0.5 * dt * (1.0 / therms[j]["Q"]) * (2 * ke - therms[j]["dof"] * utility.kB * therms[j]["T"]) *\
                            tf.math.exp(-0.25 * dt * therms[j + 1]["xi"]))
                #TODO!: REMOVE
                therms[j]["xi"] = tf.compat.v1.debugging.check_numerics(
                    therms[j]["xi"], message="therm_xi_zero")
        return therms

# returns full therms dictionary with updated eta tensor
def update_eta(therms, dt: float):
    with tf.name_scope("update_eta"):
        for j in range(0, len(therms)-1):
            if (j >= len(therms)-1): # len(therms)-1 has Q==0 which is skipped
                raise Exception("can't change last j, {}".format(j))
            therms[j]["eta"] = therms[j]["eta"] + 0.5 * dt * therms[j]["xi"]
        return therms

def calc_exp_factor(therms, dt):
    return tf.math.exp(-0.5 * dt * therms[0]["xi"])

if __name__ == "__main__":
    r = make_thremostats(5, 5)
    print(len(r),r)
    r_place, r_names = get_placeholders(r)

    print("\n\n", r_place)
    feed = therms_to_feed_dict(r, r_names)
    print(feed, "\n\n")
    x = reverse_update_xi(r_place, 0.1, 47.81)
    x = update_eta(x, 0.1)
    x = forward_update_xi(x, 0.1, 47.81)
    print("\n result", x[0]["xi"])
    print(feed)
    sess = tf.compat.v1.Session()
    sess.as_default()
    out = sess.run(x, feed_dict=feed)
    print(out, "\n")
    ft = therms_to_feed_dict(out, r_names)
    out = sess.run(x, feed_dict=ft)
    print(out)
