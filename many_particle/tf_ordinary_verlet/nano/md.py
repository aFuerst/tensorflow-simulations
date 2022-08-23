import os
import shutil
import time

import bin
import interface
import numpy as np
import particle
import tensorflow as tf
import thermostat
import utility
import velocities

import common
import energies
import forces

ke_placeholder = tf.compat.v1.placeholder(shape=(), dtype=common.tf_dtype, name="kinetic_energy_place")

def clean():
    shutil.rmtree("./output/")
    os.mkdir("./output/")

def save(i, ion_dict, therms_dict, kinetic_energy, expfac_real):
    path="./output/"
    np.savetxt(os.path.join(path, "{}-forces".format(i)), ion_dict[interface.ion_for_str])
    np.savetxt(os.path.join(path, "{}-velocities".format(i)), ion_dict[velocities.ion_vel_str])
    np.savetxt(os.path.join(path, "{}-positions".format(i)), ion_dict[interface.ion_pos_str])
    with open(os.path.join(path, "{}-thermostats".format(i)), mode="w") as f:
        for key, value in enumerate(therms_dict):
            f.write(str(key) + ":" + thermostat.to_string(value) + "\n")
    with open(os.path.join(path, "kinetic_energy"), mode="a") as f:
        f.write(str(i) + ":" + str(kinetic_energy) + "\n")
    with open(os.path.join(path, "expfac_real"), mode="a") as f:
        f.write(str(i) + ":" + str(expfac_real) + "\n")
    with open(os.path.join(path, "temp"), mode="a") as f:
        f.write(str(i) + ":" + str(2 * kinetic_energy / (thermostat._therm_constants[0]["dof"] * utility.kB)) + "\n")

def build_graph(simul_box, thermostats, ion_dict, dt:float, sample_iter):
    # TODO: get working with tf.function => faster graph execution with or without?
    ke_g = ke_placeholder
    for i in range(sample_iter):
        thermostats = thermostat.reverse_update_xi(thermostats, dt, ke_g)
        thermostats = thermostat.update_eta(thermostats, dt)
        expfac_real_g = thermostat.calc_exp_factor(thermostats, dt)
        ion_dict = velocities.update_velocity(ion_dict, dt, expfac_real_g)
        ion_dict = particle.update_position(simul_box, ion_dict, dt)
        ion_dict = forces.for_md_calculate_force(simul_box, ion_dict)
        ion_dict = velocities.update_velocity(ion_dict, dt, expfac_real_g)

        ke_g = energies.kinetic_energy(ion_dict)
        thermostats = thermostat.update_eta(thermostats, dt)
        thermostats = thermostat.forward_update_xi(thermostats, dt, ke_g)
    return thermostats, ion_dict, bin.tf_get_ion_bin_density(simul_box, ion_dict), ke_g, expfac_real_g

def loop(simul_box, thermo_g, ion_g, bin_density_g, ion_dict, tf_ion_place, thermostats, thermos_place, session, mdremote, ke_g, expfac_real_g, initial_ke):
    print("placeholders\n", tf_ion_place, "\n", thermos_place)
    print("graph\n", thermo_g, "\n", ion_g, "\n", bin_density_g, "\n", ke_g, "\n", expfac_real_g)
    planes = common.create_feed_dict((simul_box.left_plane, simul_box.tf_place_left_plane), (simul_box.right_plane, simul_box.tf_place_right_plane))
    ke_v = initial_ke
    ion_feed = common.create_feed_dict((ion_dict, tf_ion_place))
    ft = thermostat.therms_to_feed_dict(thermostats, thermos_place)
    save(0, ion_dict, thermostats, kinetic_energy=ke_v, expfac_real=1)

    for i in range(1, mdremote.steps + 1):
        feed = {**planes, **ion_feed, **ft, ke_placeholder:ke_v}
        s = time.time()
        therms_out, ion_dict_out, (pos_bin_density, neg_bin_density), ke_v, expfac_real_v = session.run(
            [thermo_g, ion_g, bin_density_g, ke_g, expfac_real_g], feed_dict=feed)
        ion_feed = common.create_feed_dict((ion_dict_out, tf_ion_place))
        ft = thermostat.therms_to_feed_dict(therms_out, thermos_place)
        save(i, ion_dict_out, therms_out, ke_v, expfac_real_v)
        positions = ion_dict_out[interface.ion_pos_str]
        if mdremote.validate:
            common.throw_if_bad_boundaries(ion_dict_out[interface.ion_pos_str], simul_box)
            if (2 * ke_v / (thermostat._therm_constants[0]["dof"] * utility.kB)) > 2:
                raise Exception("Temperature too high! was '{}'".format(2 * ke_v / (thermostat._therm_constants[0]["dof"] * utility.kB)))
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
    tf_ion_place, ion_place_copy = common.make_tf_placeholder_of_dict(ion_dict)
    thermos_place, thermo_place_copy = thermostat.get_placeholders(thermostats)
    print("tf_dict (s)", time.time() - a)

    a = time.time()
    thermostats_g, ion_dict_g, bin_density_g, ke_g, expfac_real_g = build_graph(
        simul_box, thermos_place, tf_ion_place, mdremote.timestep, mdremote.freq)
    print("graph (s)", time.time() - a)
    tot = time.time()    
    sess.run(tf.compat.v1.global_variables_initializer())
    print("var init (s)", time.time()-tot)
    d = time.time()
    loop(simul_box, thermostats_g, ion_dict_g, bin_density_g, ion_dict,
         ion_place_copy, thermostats, thermo_place_copy, sess, mdremote, ke_g, expfac_real_g, initial_ke)
    f = time.time()
    print("total run (s)", f-d)
    print("time/step", (f-d)/(mdremote.steps*mdremote.freq))
