import tensorflow as tf
import numpy as np
import time, os, shutil

import utility, bin, forces, thermostat, velocities, particle, interface, energies, common

def clean():
    shutil.rmtree("./output/")
    os.mkdir("./output/")

def save(i, thermostats, ion_dict, therms_dict, kinetic_energy, expfac_real):
    path="./output/"
    np.savetxt(os.path.join(path, "{}-forces".format(i)), ion_dict[interface.ion_for_str])
    # print("avg_for", common.magnitude_np(np.average(ion_dict[interface.ion_for_str], axis=0)), "abs_avg_for", common.magnitude_np(np.average(
    #     np.abs(ion_dict[interface.ion_for_str]), axis=0)))
    np.savetxt(os.path.join(path, "{}-velocities".format(i)), ion_dict[velocities.ion_vel_str])
    # print("avg_vel", common.magnitude_np(np.average(np.average(ion_dict[velocities.ion_vel_str], axis=0))), "abs_avg_vel", common.magnitude_np(np.average(np.average(np.abs(ion_dict[velocities.ion_vel_str]), axis=0))))
    np.savetxt(os.path.join(path, "{}-positions".format(i)), ion_dict[interface.ion_pos_str])
    with open(os.path.join(path, "{}-thermostats".format(i)), mode="w") as f:
        for key, value in therms_dict.items():
            f.write(str(key) + ":" + thermostat.to_string(value) + "\n")
    with open(os.path.join(path, "kinetic_energy"), mode="a") as f:
        f.write(str(i) + ":" + str(2 * kinetic_energy / therms_dict[0]["dof"]*utility.kB) + "\n")
    # print("expfac_real", expfac_real)
    with open(os.path.join(path, "expfac_real"), mode="a") as f:
        f.write(str(i)+":"+str(expfac_real)+"\n")

def build_graph(simul_box, thermostats, ion_dict, dt:float, initial_ke, sample_iter):
    # TODO: get working with tf.function => faster graph execution with or without?
    ke = initial_ke
    for i in range(sample_iter):
        # thermostats = thermostat.update_chain_xi(thermostats, dt, ke, start=len(thermostats)-2, end=-1)
        thermostats = thermostat.reverse_update_xi(thermostats, dt, ke)
        thermostats = thermostat.update_eta(thermostats, dt)
        expfac_real = thermostat.calc_exp_factor(thermostats, dt)
        expfac_real=np.float64(1.0)
        # print("expfac_real", expfac_real)
        ion_dict = velocities.update_velocity(ion_dict, dt, expfac_real)
        ion_dict = particle.update_position(simul_box, ion_dict, dt)
        # print("for", ion_dict[interface.ion_for_str])
        ion_dict = forces.for_md_calculate_force(simul_box, ion_dict)
        ion_dict = velocities.update_velocity(ion_dict, dt, expfac_real)

        ke = energies.kinetic_energy(ion_dict)
        thermostats = thermostat.update_eta(thermostats, dt)
        # thermostats = thermostat.update_chain_xi(thermostats, dt, ke, start=0, end=len(thermostats)-1)
        thermostats = thermostat.forward_update_xi(thermostats, dt, ke)
    return thermostats, ion_dict, bin.tf_get_ion_bin_density(simul_box, ion_dict), ke, expfac_real


def loop(simul_box, thermo_g, ion_g, bin_density_g, ion_dict, tf_ion_place, thermostats, thermos_place, session, mdremote, ke, expfac_real):
    planes = common.create_feed_dict((simul_box.left_plane, simul_box.tf_place_left_plane), (simul_box.right_plane, simul_box.tf_place_right_plane))
    # print("ion_g", ion_g)
    # print("tf_ion_place", tf_ion_place)
    # print("ion_dict", ion_dict)
    ion_feed = common.create_feed_dict((ion_dict, tf_ion_place))
    ft = thermostat.therms_to_feed_dict(thermostats, thermos_place)
    save(0, ft, ion_dict, thermostats, kinetic_energy=energies.np_kinetic_energy(ion_dict), expfac_real=0)
    check = tf.compat.v1.add_check_numerics_ops()
    # print("check", check, type(check))
    # print("ion_feed", ion_feed)
    for i in range(1, mdremote.steps+1):
        feed = {**planes, **ion_feed, **ft}
        s = time.time()
        therms_out, ion_dict_out, (pos_bin_density, neg_bin_density), ke_v = session.run(
            [thermo_g, ion_g, bin_density_g, ke], feed_dict=feed)
        # print("pos_bin_density", pos_bin_density)
        # print("neg_bin_density", neg_bin_density)
        ion_feed = common.create_feed_dict((ion_dict_out, tf_ion_place))
        # print(therms_out)
        ft = thermostat.therms_to_feed_dict(therms_out, thermos_place)
        save(i, ft, ion_dict_out, therms_out, ke_v, 1.0)
        positions = ion_dict_out[interface.ion_pos_str]
        if mdremote.validate:
            common.throw_if_bad_boundaries(ion_dict_out[interface.ion_pos_str], simul_box)
        bin.record_densities(pos_bin_density, neg_bin_density)
        print("iteration {} done".format(i))
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
    thermostats_g, ion_dict_g, bin_density_g, ke, expfac_real = build_graph(
        simul_box, thermos_place, tf_ion_place, mdremote.timestep, initial_ke, mdremote.freq)
    print("graph (s)", time.time() - a)

    tot = time.time()    
    sess.run(tf.compat.v1.global_variables_initializer())
    print("var init (s)", time.time()-tot)
    d = time.time()
    loop(simul_box, thermostats_g, ion_dict_g, bin_density_g, ion_dict,
         ion_placeholder_names, thermostats, thermo_names, sess, mdremote, ke, expfac_real)
    f = time.time()
    print("total run (s)", f-d)
    print("time/step", (f-d)/(mdremote.steps*mdremote.freq))
