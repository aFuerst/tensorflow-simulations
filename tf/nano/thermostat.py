import tensorflow as tf
import numpy as np
import utility, common

scalar_shape = ()

class Thremostat:
    def __init__(self, i, Q:float, T:float, dof:float, xi:float, eta:float, hold:float):
        self.orig = type(i) is not str 
        self.n = i if not self.orig else "therm_"+str(i)+"_"
        self.Q=Q
        self.T=T
        self.dof=dof
        self.xi=xi
        self.eta=eta
        self.hold=hold
        if self.orig:
            self.Q_place=tf.compat.v1.placeholder(dtype=common.tf_dtype, name=self.n+"Q_place", shape=scalar_shape)
            self.T_place=tf.compat.v1.placeholder(dtype=common.tf_dtype, name=self.n+"T_place", shape=scalar_shape)
            self.dof_place=tf.compat.v1.placeholder(dtype=common.tf_dtype, name=self.n+"dof_place", shape=scalar_shape)
            self.xi_place=tf.compat.v1.placeholder(dtype=common.tf_dtype, name=self.n+"xi_place", shape=scalar_shape)
            self.eta_place=tf.compat.v1.placeholder(dtype=common.tf_dtype, name=self.n+"eta_place", shape=scalar_shape)
            self.hold_place=tf.compat.v1.placeholder(dtype=common.tf_dtype, name=self.n+"hold_place", shape=scalar_shape)

    def to_feed_dict(self):
        if self.orig:
            return {self.Q_place:self.Q, self.T_place:self.T, self.dof_place:self.dof, self.xi_place:self.xi, self.eta_place:self.eta, self.hold_place:self.hold}
        else:
            return {self.n+"Q_place":self.Q, self.n+"T_place":self.T, self.n+"dof_place":self.dof, self.n+"xi_place":self.xi, self.n+"eta_place":self.eta, self.n+"hold_place":self.hold}

    def get_placeholders(self):
        return self.Q_place, self.T_place, self.dof_place, self.xi_place, self.eta_place, self.hold_place

    def run_output_to_feed(therm_ops_dict, therm_output):
        ret = {}
        for therm_op_dict_k, therm_op_dict_v in therm_ops_dict.items():
            out = therm_output[therm_op_dict_k]
            for i, value in enumerate(out):
                ret[therm_op_dict_v[i].name] = out[i]
        # print("new therm feed", ret)
        return ret
        # for t_k, t_v in therms_out.items():
        #     t_in.append(thermostat.Thremostat(t_k, *t_v))
        # ion_feed = common.create_feed_dict((ion_dict_out, tf_ion_place))

def get_placeholders(therms):
    ret = {}
    for therm in therms:
        ret[therm.n] = therm.get_placeholders()
    return ret

def therms_to_feed_dict(therms):
    feed = {}
    for therm in therms:
        # print(therm.to_feed_dict())
        feed = {**feed, **therm.to_feed_dict()}
    return feed

def make_thremostats(chain_length_real, ions_count):
    Q = 1.0 # thremostat mass
    therms = []
    i=0
    if (chain_length_real == 1):
        therms.append(Thremostat(i, 0.0, utility.T, 3* ions_count, 0.0, 0.0, 0.0))
    else:
        therms.append(Thremostat(i, Q, utility.T, 3* ions_count, 0.0, 0.0, 0.0))
        i += 1
        while (len(therms) != chain_length_real - 1):
            therms.append(Thremostat(i, Q / (3 * ions_count), utility.T, 1.0, 0.0, 0.0, 0.0))
            i += 1
        # final therms is dummy therms (dummy therms always has zero mass)
        therms.append(Thremostat(i, 0, utility.T, 3*ions_count, 0.0, 0.0, 0.0))
    return therms
  
# returns full therms dictionary with updated xi tensor
def update_chain_xi(therms, dt: float, ke):
    with tf.name_scope("update_chain_xi"):
        for j in range(len(therms)-1):
            if (j == len(therms)-1):
                raise Exception("can't change last j")
            if (j != 0):
                therms[j].xi_place = therms[j].xi_place * tf.math.exp(-0.5 * dt * therms[j + 1].xi_place) + 0.5 * dt * (1.0 / therms[j].Q_place) *\
                                                                    (therms[j - 1].Q_place * therms[j - 1].xi_place * therms[j - 1].xi_place -\
                                                                     therms[j].dof_place * utility.kB * therms[j].T_place) *\
                                                                    tf.math.exp(-0.25 * dt * therms[j + 1].xi_place)
            else:
                therms[j].xi_place = therms[j].xi_place * tf.math.exp(-0.5 * dt * therms[j + 1].xi_place) +\
                            0.5 * dt * (1.0 / therms[j].Q_place) * (2 * ke - therms[j].dof_place * utility.kB * therms[j].T_place) *\
                            tf.math.exp(-0.25 * dt * therms[j + 1].xi_place)
        return therms

# returns full therms dictionary with updated eta tensor
def update_eta(therms, dt: float):
    with tf.name_scope("update_eta"):
        for j in range(len(therms)-1):
            therms[j].eta_place = therms[j].eta + 0.5 * dt * therms[j].xi_place
        return therms

if __name__ == "__main__":
    from tensorflow_manip import silence
    silence()
    r = make_thremostats(1, 5)
    print(len(r),r)
    x = update_chain_xi(r, 1.0, 1.0)
    print("\n result", x[0].xi)
    print()
    r = make_thremostats(5, 5)
    print(len(r),r)

    feed = therms_to_feed_dict(r)
    x = update_chain_xi(r, 1.0, 1.0)
    print("\n result", x[0].xi)
    print(feed)
    sess = tf.compat.v1.Session()
    sess.as_default()
    print(sess.run(x[0].xi, feed_dict=feed))