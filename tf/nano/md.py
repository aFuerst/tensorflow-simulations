import tensorflow as tf
import numpy as np
import time, os, shutil

import utility, bin, forces, thermostat, velocities, particle, interface, energies, common

def clean():
    shutil.rmtree("./output/")
    os.mkdir("./output/")

def save(i, thermostats, ion_dict):
    path="./output/"
    np.savetxt(os.path.join(path, "{}-forces".format(i)), ion_dict[interface.ion_for_str])
    np.savetxt(os.path.join(path, "{}-velocities".format(i)), ion_dict[velocities.ion_vel_str])
    np.savetxt(os.path.join(path, "{}-positions".format(i)), ion_dict[interface.ion_pos_str])

def build_graph(simul_box, thermostats, ion_dict, dt:float, initial_ke, sample_iter):
    # TODO: get working with tf.function => faster graph execution with or without?
    ke = initial_ke
    pef = plj = lw_lj = rw_lj = erw = elw = None
    for i in range(sample_iter):
        thermostats = thermostat.update_chain_xi(thermostats, dt, ke)
        thermostats = thermostat.update_eta(thermostats, dt)
        expfac_real = thermostat.calc_exp_factor(thermostats, dt)
        # print("expfac_real", expfac_real)
        ion_dict = velocities.update_velocity(ion_dict, dt, expfac_real)
        ion_dict = particle.update_position(simul_box, ion_dict, dt)
        # print("for", ion_dict[interface.ion_for_str])
        ion_dict = forces.for_md_calculate_force(simul_box, ion_dict)
        ion_dict = velocities.update_velocity(ion_dict, dt, expfac_real)

        ke = energies.kinetic_energy(ion_dict)
        thermostats = thermostat.update_eta(thermostats, dt)
        thermostats = thermostat.update_chain_xi(thermostats, dt, ke)
    return thermostats, ion_dict, bin.tf_get_ion_bin_density(simul_box, ion_dict)

def loop(simul_box, thermo_g, ion_g, bin_density_g, ion_dict, tf_ion_place, thermostats, thermos_place, session, mdremote):
    planes = common.create_feed_dict((simul_box.left_plane, simul_box.tf_place_left_plane), (simul_box.right_plane, simul_box.tf_place_right_plane))
    # print("ion_g", ion_g)
    # print("tf_ion_place", tf_ion_place)
    # print("ion_dict", ion_dict)
    ion_feed = common.create_feed_dict((ion_dict, tf_ion_place))
    ft = thermostat.therms_to_feed_dict(thermostats, thermos_place)
    save(0, ft, ion_dict)
    # print("ion_feed", ion_feed)
    for i in range(1, mdremote.steps+1):
        feed = {**planes, **ion_feed, **ft}
        s = time.time()
        therms_out, ion_dict_out, (pos_bin_density, neg_bin_density) = session.run([thermo_g, ion_g, bin_density_g], feed_dict=feed)
        if mdremote.validate:
            common.throw_if_bad_boundaries(ion_dict_out[interface.ion_pos_str], simul_box)
        print("pos_bin_density", pos_bin_density)
        print("neg_bin_density", neg_bin_density)
        ion_feed = common.create_feed_dict((ion_dict_out, tf_ion_place))
        # print(therms_out)
        ft = thermostat.therms_to_feed_dict(therms_out, thermos_place)
        save(i, ft, ion_dict_out)
        pos_bin_density, neg_bin_density = session.run([bin_density_g], feed_dict=feed)
        bin.record_densities(pos_bin_density, neg_bin_density)
    bin.get_density_profile()

def run_md_sim(simul_box, thermostats, ion_dict, charge_meshpoint: float, valency_counterion: int, mdremote):
    clean()
    ion_dict = forces.initialize_forces(ion_dict)
    initial_ke = energies.np_kinetic_energy(ion_dict)

    sess = tf.compat.v1.Session()
    sess.as_default()

    a = time.time()
    tf_ion_place, ion_placeholder_names = common.make_tf_placeholder_of_dict(ion_dict)
    thermos_place, thermo_names = thermostat.get_placeholders(thermostats)
    print("tf_dict (s)", time.time() - a)

    a = time.time()
    thermostats_g, ion_dict_g, bin_density_g, = build_graph(simul_box, thermos_place, tf_ion_place, mdremote.timestep, initial_ke, mdremote.freq)
    print("graph (s)", time.time() - a)

    tot = time.time()    
    sess.run(tf.compat.v1.global_variables_initializer())
    print("var init (s)", time.time()-tot)
    d = time.time()
    loop(simul_box, thermostats_g, ion_dict_g, bin_density_g, ion_dict, ion_placeholder_names, thermostats, thermo_names, sess, mdremote)
    f = time.time()
    print("total run (s)", f-d)
    print("time/step", (f-d)/(mdremote.steps*mdremote.freq))