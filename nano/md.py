import tensorflow as tf
import time, os, shutil, datetime
import utility, bin, forces, thermostat, velocities, particle, interface, energies, common, control
import tensorboard

ke_placeholder = tf.compat.v1.placeholder(shape=(), dtype=common.tf_dtype, name="kinetic_energy_place")

def run_md_sim(simul_box, thermostats, ion_dict, charge_meshpoint, valency_counterion: int, mdremote, bins, tf_sess_config):
    # if os.path.exists("output/logs/"):
    #     shutil.rmtree("output/logs/")
    print("\n os.cpu_count():", str(os.cpu_count()))
    initial_ke = energies.np_kinetic_energy(ion_dict)
    print("initial KE:", initial_ke)
    # config = tf.compat.v1.ConfigProto()
    # config.intra_op_parallelism_threads = 1
    # config.inter_op_parallelism_threads = 1
    # tf.session(config=config)
    sess = tf.compat.v1.Session(config=tf_sess_config)
    sess.as_default()
    a = time.time()
    # TODO: better way to initialize forces, without using eval(), maybe initialize using numpy version.
    #  This is because if force is fed to session as an unevaluated tensor, the code throws an exception
    ion_dict[interface.ion_for_str] = (forces.for_md_calculate_force(simul_box, ion_dict, charge_meshpoint))[
        interface.ion_for_str].eval(session=sess)
    tf_ion_place, ion_place_copy = common.make_tf_placeholder_of_dict(ion_dict)
    thermos_place, thermo_place_copy = thermostat.get_placeholders(thermostats)
    thermostats_g, ion_dict_g, ke_g, expfac_real_g, pe_g, bath_ke_g, bath_pe_g, pos_bin_density_g, neg_bin_density_g = build_graph(
        simul_box, thermos_place, tf_ion_place, mdremote.timestep, mdremote, bins, charge_meshpoint)
    sess.run(tf.compat.v1.global_variables_initializer())
    loop(pe_g, bath_ke_g, bath_pe_g, simul_box, thermostats_g, ion_dict_g, ion_dict,
         ion_place_copy, thermostats, thermo_place_copy, sess, mdremote, ke_g, expfac_real_g, initial_ke, bins,
         pos_bin_density_g, neg_bin_density_g)


# def clean():
#     if os.path.exists("output/"):
#         shutil.rmtree("output/")
#     if not os.path.exists("output/"):
#         os.mkdir("output/")

def save(i, ion_dict, therms_dict, kinetic_energy, expfac_real, mydir):
    with open(os.path.join(mydir, 'forces.dat'), 'a') as force_file:
        force_file.write(str(i)+"\t"+str(ion_dict[interface.ion_for_str][0][0]) + "\t" + str(ion_dict[interface.ion_for_str][0][1]) + "\t" + str(ion_dict[interface.ion_for_str][0][2])+"\t")
        force_file.write(str(ion_dict[interface.ion_for_str][1][0]) + "\t" + str(
            ion_dict[interface.ion_for_str][1][1]) + "\t" + str(ion_dict[interface.ion_for_str][1][2]) + "\n")
    with open(os.path.join(mydir, 'velocities.dat'), 'a') as velocity_file:
        velocity_file.write(str(i)+"\t"+str(ion_dict[velocities.ion_vel_str][0][0]) + "\t" + str(ion_dict[velocities.ion_vel_str][0][1]) + "\t" + str(ion_dict[velocities.ion_vel_str][0][2]) + "\n")
    with open(os.path.join(mydir, "postions.dat"), "a") as position_file:
        position_file.write(str(i)+"\t"+ str(ion_dict[interface.ion_pos_str][0][0]) + "\t" + str(ion_dict[interface.ion_pos_str][0][1]) + "\t" + str(ion_dict[interface.ion_pos_str][0][2]) +"\t"+ str(ion_dict[interface.ion_pos_str][1][0]) + "\t" + str(ion_dict[interface.ion_pos_str][1][1]) + "\t" + str(ion_dict[interface.ion_pos_str][1][2]) + "\n")
    with open(os.path.join(mydir, "bathxi.dat"), mode="a") as bathxi_file:
        bathxi_file.write(str(i) + "\t" + str(therms_dict[0]["xi"]) + "\t" + str(therms_dict[1]["xi"]) + "\t" + str(therms_dict[2]["xi"]) + "\t" + str(therms_dict[3]["xi"]) + "\t" + str(therms_dict[4]["xi"]) + "\n") #+ therms_dict[4]["xi"] + "\t")
    with open(os.path.join(mydir, "batheta.dat"), mode="a") as batheta_file:
        batheta_file.write(str(i) + "\t" + str(therms_dict[0]["eta"]) + "\t" + str(therms_dict[1]["eta"]) + "\t" + str(therms_dict[2]["eta"]) + "\t" + str(therms_dict[3]["eta"]) + "\t" + str(therms_dict[4]["eta"]) + "\n") #+ therms_dict[4]["xi"] + "\t")
    with open(os.path.join(mydir, "expfac_real"), mode="a") as f:
        f.write(str(i) + "\t" + str(expfac_real) + "\n")

def build_graph(simul_box, thermostats, ion_dict, dt:float, mdremote, bins, charge_meshpoint):
    # with tf.name_scope("velocity_verlet"):
    # TODO: get working with tf.function => faster graph execution with or without?
    ke_g = ke_placeholder
    for i in range(0, mdremote.freq):
        # print("inside for loop of build graph")
        thermostats = thermostat.reverse_update_xi(thermostats, dt, ke_g)
        thermostats = thermostat.update_eta(thermostats, dt)
        expfac_real_g = thermostat.calc_exp_factor(thermostats, dt)
        ion_dict = velocities.update_velocity(ion_dict, dt, expfac_real_g)
        ion_dict = particle.update_position(simul_box, ion_dict, dt)
        ion_dict = forces.for_md_calculate_force(simul_box, ion_dict, charge_meshpoint)
        ion_dict = velocities.update_velocity(ion_dict, dt, expfac_real_g)
        ke_g = energies.kinetic_energy(ion_dict)
        thermostats = thermostat.update_eta(thermostats, dt)
        thermostats = thermostat.forward_update_xi(thermostats, dt, ke_g)
    (pos_bin_density, neg_bin_density) = bin.tf_get_ion_bin_density(simul_box, ion_dict, bins)
    pe_g = energies.energy_functional(simul_box, charge_meshpoint, ion_dict)
    bath_ke_g = energies.bath_kinetic_energy(thermostats)
    bath_pe_g = energies.bath_potential_energy(thermostats)
    return thermostats, ion_dict, ke_g, expfac_real_g, pe_g, bath_ke_g, bath_pe_g, pos_bin_density, neg_bin_density

def save_useful_data(i, particle_ke, potential_energy, real_bath_ke, real_bath_pe, path):
    f_therms_file = open(os.path.join(path,"temp.dat"), 'a')
    f_energy_file = open(os.path.join(path,"energy.dat"), 'a')
    f_therms_file.write(str(i)+"\t"+str(2*particle_ke/(thermostat._therm_constants[0]["dof"]*utility.kB))+"\t"+str(thermostat._therm_constants[0]["T"])+"\t"+str(thermostat._therm_constants[0]["Q"])+"\n")
    ext_energy = particle_ke+potential_energy+real_bath_ke+real_bath_pe
    f_energy_file.write(
        str(i) + "\t" + str(ext_energy) + "\t" + str(particle_ke) + "\t" + str(
            potential_energy) + "\t" + str(
            real_bath_ke) + "\t" + str(
            real_bath_pe) + "\n")
    # f_energy_file.write(str(i)+"\t"+str(ext_energy.eval(session=tf.compat.v1.Session()))+"\t"+str(particle_ke)+"\t"+str(potential_energy.eval(session=tf.compat.v1.Session()))+"\t"+str(real_bath_ke.eval(session=tf.compat.v1.Session()))+"\t"+str(real_bath_pe.eval(session=tf.compat.v1.Session()))+"\n")
    # TODO : eval() takes more time than np.savetxt(). find a better way to print tensors
    f_energy_file.close()


def loop(pe_g, bath_ke_g, bath_pe_g, simul_box, thermo_g, ion_g, ion_dict, tf_ion_place, thermostats, thermos_place, session, mdremote, ke_g, expfac_real_g, initial_ke, bins, pos_bin_density_g, neg_bin_density_g):

    profile = True
    planes = common.create_feed_dict((simul_box.left_plane, simul_box.tf_place_left_plane), (simul_box.right_plane, simul_box.tf_place_right_plane))
    ke_v = initial_ke
    ion_feed = common.create_feed_dict((ion_dict, tf_ion_place))
    ft = thermostat.therms_to_feed_dict(thermostats, thermos_place)
    print("\n Running MD Simulation for ",mdremote.steps," steps")
    writer = tf.compat.v1.summary.FileWriter("output/logs/")
    writer.add_graph(session.graph)
    if profile:
        run_options = tf.compat.v1.RunOptions(trace_level=tf.compat.v1.RunOptions.FULL_TRACE)
        run_metadata = tf.compat.v1.RunMetadata()
    for i in range(1, (mdremote.steps//mdremote.freq + 1)):
        feed = {**planes, **ion_feed, **ft, ke_placeholder: ke_v}
        s = time.time()
        therms_out, ion_dict_out, ke_v, pe_v, bath_ke_v, bath_pe_v, expfac_real_v, pos_bin_density_v, neg_bin_density_v = session.run([thermo_g, ion_g, ke_g, pe_g, bath_ke_g, bath_pe_g, expfac_real_g, pos_bin_density_g, neg_bin_density_g], feed_dict=feed, options=run_options, run_metadata=run_metadata)
        ion_feed = common.create_feed_dict((ion_dict_out, tf_ion_place))
        ft = thermostat.therms_to_feed_dict(therms_out, thermos_place)
        if mdremote.validate:
            print("\n Entered Validation:::")
            common.throw_if_bad_boundaries(ion_dict_out[interface.ion_pos_str], simul_box)
            if (2 * ke_v / (thermostat._therm_constants[0]["dof"] * utility.kB)) > 2:
                raise Exception("Temperature too high! was '{}'".format(2 * ke_v / (thermostat._therm_constants[0]["dof"] * utility.kB)))
        if profile:
            from tensorflow.python.client import timeline
            tl = timeline.Timeline(run_metadata.step_stats)
            ctf = tl.generate_chrome_trace_format()
            with open(os.path.join("output/logs/", "{}-profile_timeline.json".format(i)), 'w') as f:
                f.write(ctf)


        # compute_n_write_useful_data
        # if i >= 0 and (i*mdremote.freq) % mdremote.extra_compute == 0:
        #    save_useful_data(i, ke_v, pe_v, bath_ke_v, bath_pe_v, utility.root_path)
        print("iteration {} done".format(i))

        # generate movie file
        # moviestart = 1
        # if i >= moviestart and i % mdremote.moviefreq == 0:
        #     make_movie(i, ion_dict_out, simul_box)

        # Write density profile
        #if i >= mdremote.hiteqm and i%mdremote.writedensity==0: #if (i*mdremote.freq) >= mdremote.hiteqm:
        # if i==5000000:
        #     bin.record_densities(i, pos_bin_density_v, neg_bin_density_v, i//mdremote.writedensity, bins, mdremote.writedensity)
        # TODO : average_errorbars_density()
    print("\n Simulation ends!!")

def make_movie(num, ion, box):
    path = utility.root_path
    f_movie_file = open(os.path.join(path, "electrolyte_movie.xyz"), 'a')
    f_movie_file.write("ITEM: TIMESTEP \n")
    f_movie_file.write(str(num-1)+"\n")
    f_movie_file.write("ITEM: NUMBER OF ATOMS \n")
    f_movie_file.write(str(ion[interface.ion_valency_str].size)+"\n")
    f_movie_file.write("ITEM: BOX BOUNDS \n")
    f_movie_file.write(str(-0.5*box.lx)+"\t"+str(0.5*box.lx)+"\n")
    f_movie_file.write(str(-0.5 * box.ly) + "\t" + str(0.5 * box.ly)+"\n")
    f_movie_file.write(str(-0.5 * box.lz) + "\t" + str(0.5 * box.lz)+"\n")
    f_movie_file.write("ITEM: ATOMS index type q x y z \n")
    condition = tf.greater(ion[interface.ion_valency_str], 0)
    type = tf.compat.v1.where_v2(condition, 1, -1)
    for i in range(0, ion[interface.ion_valency_str].size):
        f_movie_file.write(str(i)+"\t"+str(type[i].eval(session=tf.compat.v1.Session()))+"\t"+str(ion[interface.ion_valency_str][i])+"\t"+str(ion[interface.ion_pos_str][i][0])+"\t"+str(ion[interface.ion_pos_str][i][1])+"\t"+str(ion[interface.ion_pos_str][i][2])+"\n")
    f_movie_file.close()
    pass
