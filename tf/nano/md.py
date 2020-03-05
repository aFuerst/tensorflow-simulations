import tensorflow as tf
import numpy as np
import time 

import utility, bin, forces, thermostat, velocities, particle, interface, energies, common

def build_graph(simul_box, thermostats, ion_dict, dt:float, initial_ke):
    # TODO: get working with tf.function
    ke = initial_ke
    for i in range(100):
        thermostats = thermostat.update_chain_xi(thermostats, dt, ke)
        thermostats = thermostat.update_eta(thermostats, dt)
        expfac_real = tf.math.exp(-0.5 * dt * thermostats[0].xi_place)

        ion_dict = velocities.update_velocity(ion_dict, dt, expfac_real)
        ion_dict = particle.update_position(simul_box, ion_dict, dt)
        ion_dict = forces.for_md_calculate_force(simul_box, ion_dict)
        ion_dict = velocities.update_velocity(ion_dict, dt, expfac_real)

        ke = energies.kinetic_energy(ion_dict)
        thermostats = thermostat.update_eta(thermostats, dt)
        thermostats = thermostat.update_chain_xi(thermostats, dt, ke)
    return thermostats, ion_dict, bin.tf_get_ion_bin_density(simul_box, ion_dict)

def loop(simul_box, thermo_g, ion_g, bin_density_g, ion_dict, tf_ion_place, thermostats, session, iterations):
    planes = common.create_feed_dict((simul_box.left_plane, simul_box.tf_place_left_plane), (simul_box.right_plane, simul_box.tf_place_right_plane))
    ion_feed = common.create_feed_dict((ion_dict, tf_ion_place))
    ft = thermostat.therms_to_feed_dict(thermostats)
    for i in range(iterations):
        feed = {**planes, **ion_feed, **ft}
        s = time.time()
        therms_out, ion_dict_out, bin_density = session.run([thermo_g, ion_g, bin_density_g], feed_dict=feed)
        print(bin_density)
        print("part run", i, time.time()-s)
        ion_feed = common.create_feed_dict((ion_dict_out, tf_ion_place))
        ft = thermostat.Thremostat.run_output_to_feed(thermo_g, therms_out)

def run_md_sim(args, simul_box, thermostats, ion_dict, bins, charge_meshpoint: float, valency_counterion: int, mdremote):
    ion_dict = forces.initialize_forces(ion_dict)
    initial_ke = energies.np_kinetic_energy(ion_dict)

    sess = tf.compat.v1.Session()
    sess.as_default()

    a = time.time()
    tf_ion_place = common.make_tf_placeholder_of_dict(ion_dict)
    print("tf_dict", time.time() - a)

    a = time.time()
    thermostats_g, ion_dict_g, bin_density_g = build_graph(simul_box, thermostats, tf_ion_place, mdremote.timestep, initial_ke)
    thermostats_g = thermostat.get_placeholders(thermostats_g)
    print("graph", time.time() - a)

    tot = time.time()    
    sess.run(tf.compat.v1.global_variables_initializer())
    print("var init", time.time()-tot)
    d = time.time()
    loop(simul_box, thermostats_g, ion_dict_g, bin_density_g, ion_dict, tf_ion_place, thermostats, sess, 1000)
    print("total run", time.time()-d)

