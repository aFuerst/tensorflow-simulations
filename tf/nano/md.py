import tensorflow as tf
import numpy as np
import time 

import utility, bin, forces, thermostat, velocities, particle, interface, energies, common

# @tf.function
def build_graph(simul_box, thermostats, ion_dict, dt:float, initial_ke):
    ke = initial_ke
    for i in range(100):
        thermostats = thermostat.update_chain_xi(thermostats, dt, initial_ke)
        thermostats = thermostat.update_eta(thermostats, dt)
        expfac_real = tf.math.exp(-0.5 * dt * thermostats[0].xi_place)

        ion_dict = velocities.update_velocity(ion_dict, dt, expfac_real)
        ion_dict = particle.update_position(simul_box, ion_dict, dt)
        ion_dict = forces.for_md_calculate_force(simul_box, ion_dict)
        ion_dict = velocities.update_velocity(ion_dict, dt, expfac_real)

        ke = energies.kinetic_energy(ion_dict)
        thermostats = thermostat.update_eta(thermostats, dt)
        thermostats = thermostat.update_chain_xi(thermostats, dt, ke)
    return thermostats, ion_dict

def run_md_sim(simul_box, thermostats, ion_dict, bins, charge_meshpoint: float, valency_counterion: int, mdremote):
    ion_dict = forces.initialize_forces(ion_dict)
    initial_ke = energies.np_kinetic_energy(ion_dict)

    sess = tf.compat.v1.Session()
    sess.as_default()

    a = time.time()
    tf_ion_real, tf_ion_place = common.make_tf_versions_of_dict(ion_dict)
    thermostats_g, ion_dict_g = build_graph(simul_box, thermostats, tf_ion_place, mdremote.timestep, initial_ke)
    print(thermostats_g)
    thermostats_g = thermostat.to_fetch(thermostats_g)
    b = time.time()

    c = time.time()
    feed = common.create_feed_dict((simul_box.left_plane, simul_box.tf_place_left_plane), (simul_box.right_plane, simul_box.tf_place_right_plane), (ion_dict, tf_ion_place))
    ft = thermostat.therms_to_feed_dict(thermostats)
    feed = {**feed, **ft}
    sess.run(tf.compat.v1.global_variables_initializer())
    
    result = sess.run([thermostats_g, ion_dict_g], feed_dict=feed)
    d = time.time()
    print(result)

    print("graph", b - a)
    print("run", d - c)
    # bin_nums_tf = bin.tf_get_ion_bin_density(simul_box, ion_p, bins) # build 'graph'
    # feed_dict = {ion_p:ion_dict[interface.ion_pos_str]}
    # bin_nums = sess.run(bin_nums_tf, feed_dict=feed_dict)
    # print(bin_nums)
