import tensorflow as tf
import numpy as np
import argparse, math, os, datetime
import logging
import utility, control, interface, bin, thermostat, md, velocities, forces, common
import tensorflow_manip as tfmanip

np.random.seed(0)  # be consistent


# import sys
# np.set_printoptions(threshold=sys.maxsize)
class MdSimulation:
    def __init__(self):
        self.input_systems = []
        self.concentration = 0.5
        self.pos_diameter = 0.714
        self.neg_diameter = 0.714
        self.pos_valency = 1
        self.neg_valency = -1
        self.confinment_len = 3

    def start_sim(self, tf_sess_config, args, logger):
        utility.root_path = os.path.join("output/simulation/", utility.system_params)
        if not os.path.exists(utility.root_path):
            os.makedirs(utility.root_path)

        print("DEBUG:", self.neg_diameter, self.concentration, self.confinment_len)
        negative_diameter_in = self.neg_diameter
        positive_diameter_in = self.pos_diameter
        charge_density = args.charge_density
        if (positive_diameter_in <= negative_diameter_in):
            utility.unitlength = positive_diameter_in
            smaller_ion_diameter = positive_diameter_in
            bigger_ion_diameter = negative_diameter_in
        else:
            utility.unitlength = negative_diameter_in
            smaller_ion_diameter = negative_diameter_in
            bigger_ion_diameter = positive_diameter_in

        utility.unittime = math.sqrt(utility.unitmass * utility.unitlength * pow(10.0, -7) * utility.unitlength / utility.unitenergy)
        utility.scalefactor = utility.epsilon_water * utility.lB_water / utility.unitlength
        bz = self.confinment_len
        salt_conc_in = self.concentration
        # bx = math.sqrt(212 / 0.6022 / salt_conc_in / bz)
        bx = args.bx
        if salt_conc_in > 0.5:
            bx = 10.0
            args.fraction_diameter = 1 / 28.0
        else:
            bx = 15.0
            args.fraction_diameter = 1 / 42.0
        by = bx
        # print("In nanometers, bx:",str(bx), " by: ", str(by)," bz:", bz, "\nunit length:", utility.unitlength, "\nsal conc:",salt_conc_in)
        if (charge_density < -0.01 or charge_density > 0.0):  # we can choose charge density on surface between 0.0 (uncharged surfaces)  to -0.01 C/m2.
            logger.info("charge density on the surface must be between zero to -0.01 C/m-2 aborting")
            exit(1)
        pz_in = self.pos_valency
        valency_counterion = 1  # pz_in
        counterion_diameter_in = positive_diameter_in
        surface_area = bx * by * pow(10.0, -18)  # in unit of squared meter
        number_meshpoints = pow((1.0 / args.fraction_diameter), 2.0)
        charge_meshpoint = (charge_density * surface_area) / (utility.unitcharge * number_meshpoints)
        # in unit of electron charge
        total_surface_charge = charge_meshpoint * number_meshpoints  # in unit of electron charge
        counterions = 2.0 * (int(abs(total_surface_charge) / valency_counterion))  # there are two charged surfaces, we multiply the counter ions by two
        nz_in = self.neg_valency

        # we should make sure the total charge of both surfaces and the counter ions are zero
        if (((valency_counterion * counterions) + (total_surface_charge * 2.0)) != 0):
            # we distribute the extra charge to the mesh points to make the system electroneutral then we recalculate the charge density on surface
            charge_meshpoint = -1.0 * (valency_counterion * counterions) / (number_meshpoints * 2.0)
            total_surface_charge = charge_meshpoint * number_meshpoints  # we recalculate the total charge on teh surface
            charge_density = (total_surface_charge * utility.unitcharge) / surface_area  # in unit of Coulomb per squared meter

        mdremote = control.Control(args)

        if (mdremote.steps < 100000):  # minimum mdremote.steps is 20000
            mdremote.hiteqm = int(mdremote.steps * 0.1)
            # mdremote.writedensity = int(mdremote.steps * 0.1)
            # mdremote.extra_compute = int(mdremote.steps * 0.01)
            # mdremote.moviefreq = int(mdremote.steps * 0.001)
        else:
            mdremote.hiteqm = int(mdremote.steps * 0.2)
            # mdremote.writedensity = int(mdremote.steps * 0.1)
            # mdremote.extra_compute = int(mdremote.steps * 0.01)
            # mdremote.moviefreq = int(mdremote.steps * 0.001)



        T = 1
        simul_box = interface.Interface(salt_conc_in=salt_conc_in, salt_conc_out=0, salt_valency_in=pz_in,
                                        salt_valency_out=0, bx=bx / utility.unitlength, by=by / utility.unitlength,
                                        bz=bz / utility.unitlength, \
                                        initial_ein=mdremote.ein, initial_eout=mdremote.eout)
        ion_dict = simul_box.put_saltions_inside(logger, pz=pz_in, nz=nz_in, concentration=salt_conc_in,
                                                 positive_diameter_in=positive_diameter_in, \
                                                 negative_diameter_in=negative_diameter_in, counterions=counterions,
                                                 valency_counterion=valency_counterion, \
                                                 counterion_diameter_in=counterion_diameter_in,
                                                 bigger_ion_diameter=bigger_ion_diameter, crystal_pack=args.random_pos_init)
        bins = bin.Bin().make_bins(simul_box, args.bin_width, ion_dict[interface.ion_diameters_str][0])
        (pos_bin_density, neg_bin_density) = bin.Bin().bin_ions(simul_box, ion_dict, bins)
        # pos_bin_density+neg_bin_density
        simul_box.discretize(smaller_ion_diameter / utility.unitlength, args.fraction_diameter, charge_meshpoint)

        # write initial densities
        density_pos = "initial_density_pos.dat"
        density_neg = "initial_density_neg.dat"
        f_den_pos = open(os.path.join(utility.root_path, density_pos), 'w')
        f_den_neg = open(os.path.join(utility.root_path, density_neg), 'w')
        pos_numpy = pos_bin_density.eval(session=tf.compat.v1.Session())
        neg_numpy = neg_bin_density.eval(session=tf.compat.v1.Session())
        for b in range(0, len(bins)):
            f_den_pos.write(str(bins[b].midpoint)+"\t"+str(pos_numpy[b])+"\n")
            f_den_neg.write(str(bins[b].midpoint)+"\t"+str(neg_numpy[b])+"\n")
        f_den_neg.close()
        f_den_pos.close()

        #setup thermostats
        thermos = thermostat.make_thermostats(args.chain_length_real, ions_count=len(ion_dict[interface.ion_pos_str]),
                                              Q=args.therm_mass)

        ion_dict = velocities.initialize_particle_velocities(ion_dict, thermos)
        ion_dict = forces.for_md_calculate_force(simul_box, ion_dict, charge_meshpoint)

        #np.power is just used for conversion
        # pos_ions = pos_bin_density * np.power(bins[0].volume, 1)
        # neg_ions = neg_bin_density * np.power(bins[0].volume, 1)

        # printing system information
        logger.info("Total_surface_charge:"+str(total_surface_charge)+"\ncharge_meshpoint:"+str(charge_meshpoint))
        logger.info("Charge density:"+str(charge_density))
        logger.info("Dielectric constant of water "+str(utility.epsilon_water))
        logger.info("Unit length "+ str(utility.unitlength))
        logger.info("Unit of mass "+ str(utility.unitmass))
        logger.info("Unit of energy "+ str(utility.unitenergy))
        logger.info("Unit time "+ str(utility.unittime))
        logger.info("Simulation box dimensions (in reduced units) x | y | z "+str(simul_box.lx)+"|"+str(simul_box.ly)+"|"+str(simul_box.lz))
        logger.info("Box dimensions (in nanometers) x | y | z "+str(bx)+"|"+str(by)+"|"+str(bz))
        logger.info("Permittivity inside and outside the confinement (channel) "+ str(mdremote.ein)+","+ str(mdremote.eout))
        logger.info("Dielectric contrast across interfaces "+ str(2 * (mdremote.eout - mdremote.ein) / (mdremote.eout + mdremote.ein)))
        logger.info("Positive (+) ion valency "+str(pz_in))
        logger.info("Negative (-) ion valency "+ str(nz_in))
        logger.info("Valency of counter ions is "+str(valency_counterion))
        logger.info("positive ion diameter (red. units) "+str(positive_diameter_in/utility.unitlength))
        logger.info("negative ion diameter (red. units) "+str(negative_diameter_in / utility.unitlength))
        logger.info("counter ion diameter (red units) "+str(counterion_diameter_in / utility.unitlength))
        logger.info("In MD, charge density "+str(charge_density))
        logger.info("Ion (salt) concentration (c) inside "+ str(salt_conc_in)+ " M")
        logger.info("Debye length "+ str(simul_box.inv_kappa_in))
        logger.info("Mean separation between ions " + str(simul_box.mean_sep_in))
        logger.info("Temperature (in Kelvin) " + str(utility.room_temperature))
        logger.info("Binning width (uniform) " + str(bins[0].width))
        logger.info("Number of bins" +str(len(bins)))
        logger.info("Number of points discretizing the left and right planar walls/interfaces/surfaces " + str(len(simul_box.left_plane["posvec"]))+" " + str(len(simul_box.right_plane["posvec"])))
        logger.info("Number of ions "+ str(len(ion_dict[interface.ion_pos_str])+counterions))
        # print("Number of positive ions ", pos_ions)
        # print("Number of negative ions ", neg_ions)
        logger.info("Number of counter ions " + str(counterions))
        logger.info("Time step in the simulation" + str(mdremote.timestep))

        md.run_md_sim(logger, tf_sess_config, simul_box, thermos, ion_dict, charge_meshpoint, valency_counterion, mdremote, bins)
        # Starting screen factor processing
        # if charge_density != 0:
        #     screen = True
        #     print("Screen Factor Processing Started.")
        #     number_of_bins = simul_box.lz//bins[0].width
        #     bin_width = simul_box/number_of_bins
        #     bin_width = bin_width * 0.01
        #     cnt_filename = 0
        #     samples = 0
        #     screen_bins = bin.Bin().make_bins(simul_box, bin_width,ion_dict[interface.ion_diameters_str][0])


    # if __name__ == "__main__":
    def start(self, data):
        # self.read_input(filename)
        print("***** Starting MD Simulations for randomly sampled systems to prepare training data *****")
        print('{0: <50}'.format("Simulating the following systems one by one"))
        parser = argparse.ArgumentParser()
        parser.add_argument('-c', "--cpu", action="store_true")
        parser.add_argument('-v', "--verbose", action="store_true")
        parser.add_argument('-x', "--xla", action="store_true")
        parser.add_argument('-r', "--prof", action="store_true")
        parser.add_argument('-o', "--opt", action="store_true")
        parser.add_argument('-d', "--charge-density", action="store", default=-0.0, type=float)
        parser.add_argument("--ein", action="store", default=80, type=float)
        parser.add_argument("--eout", action="store", default=80, type=float)
        parser.add_argument('-ec', "--extracompute", action="store", default=100000, type=int)
        parser.add_argument('-mf', "--moviefreq", action="store", default=10000, type=int)
        parser.add_argument('-he', "--hiteqm", action="store", default=100000, type=int)
        parser.add_argument('-t', "--delta-t", action="store", default=0.001, type=float)
        parser.add_argument('-s', "--steps", action="store", default=1000000, type=int)
        parser.add_argument('-f', "--freq", action="store", default=100000, type=int)
        parser.add_argument('-cf', "--cppfreq", action="store", default=100, type=int)
        parser.add_argument('-wd', "--writedensity", action="store", default=100000, type=int)
        parser.add_argument('-th', "--threads", action="store", default=os.cpu_count(), type=int)
        parser.add_argument("--validate", action="store_true")
        parser.add_argument("--random-pos-init", action="store_false")
        parser.add_argument('-bw', "--bin_width", action="store", default=0.05, type=float)  #this is in reduced units to fix it
        parser.add_argument('-fd', "--fraction_diameter", action="store", default=1/28.0, type=float)
        parser.add_argument('-chl', "--chain_length_real", action="store", default=5, type=float)
        parser.add_argument('-Q', "--therm_mass", action="store", default=1, type=float)
        parser.add_argument('--bx', action="store", default=15.3153, type=float)
        parser.add_argument('--by', action="store", default=15.3153, type=float)
        # conf_arg = parser.add_argument('-cl', "--confinment-len", action="store", default=self.conf_len, type=float)
        # pos_arg = parser.add_argument('-e', "--pos-valency", action="store", default=1, type=int)
        # neg_arg = parser.add_argument('-en', "--neg-valency", action="store", default=-1, type=int)
        # conc_arg = parser.add_argument('-M', "--concentration", action="store", default=0.5, type=float)
        # pos_diam_arg = parser.add_argument('-pd', "--pos-diameter", action="store", default=0.714, type=float)
        # neg_diam_arg = parser.add_argument('-nd', "--neg-diameter", action="store", default=0.714, type=float)
        args = parser.parse_args()
        print('{0: <20}'.format('CONFINEMENT_LEN') + '{0: <20}'.format('POS_VAL') + '{0: <20}'.format('NEG_VAL')
              + '{0: <20}'.format('CONCENTRATN') + '{0: <20}'.format('POS_DIAM') + '{0: <20}'.format('NEG_DIAM'))
        for ele in data:
            self.confinment_len = ele[0]
            self.pos_valency = ele[1]
            self.neg_valency = ele[2]
            self.concentration = ele[3]
            self.pos_diameter = ele[4]
            self.neg_diameter = ele[4]

            # parser.add_argument('-cl', "--confinment-len", action="store", default=ele[0], type=float)
            # parser.add_argument('-e', "--pos-valency", action="store", default=ele[1], type=int)
            # parser.add_argument('-en', "--neg-valency", action="store", default=ele[2], type=int)
            # parser.add_argument('-M', "--concentration", action="store", default=ele[3], type=float)
            # parser.add_argument('-pd', "--pos-diameter", action="store", default=ele[4], type=float)
            # parser.add_argument('-nd', "--neg-diameter", action="store", default=ele[4], type=float)
            # conf_arg.default = ele[0]
            # pos_arg.default = ele[1]
            # neg_arg.default = ele[2]
            # conc_arg.default = ele[3]
            # pos_diam_arg.default = ele[4]
            # neg_diam_arg.default = ele[4]
            # print("DEBUG::", conc_arg.default, ",", pos_diam_arg.default, ",", conc_arg.default)
            print('{0: <20}'.format(ele[0]) + '{0: <20}'.format(ele[1]) + '{0: <20}'.format(ele[2])
                  + '{0: <20}'.format(ele[3]) + '{0: <20}'.format(ele[4]) + '{0: <20}'.format(ele[4]))

            logging.basicConfig(filename=utility.log_path+"logs_"+datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')+".log",
                                format='%(asctime)s %(message)s',
                                filemode='w')
            logger = logging.getLogger()
            logger.setLevel(logging.DEBUG)
            tfmanip.toggle_xla(args.xla)
            tfmanip.manual_optimizer(args.opt)
            config = tfmanip.toggle_cpu(args.cpu, args.threads)
            tfmanip.silence()
            utility.system_params = "_".join(map(str, ele))  #self.conf_len+"_"+self.pos+"_"+self
            self.start_sim(config, args, logger)
        print("***** Finished MD Simulation for "+ str(len(data))+" input systems. *****")


if __name__ == "__main__":
    MdSimulation().start([[3.0, 1.0, -1.0, 0.3, 0.55],[4.0, 1.0, -1.0, 0.4, 0.55],[3.0, 1.0, -1.0, 0.4, 0.65],[4.0, 1.0, -1.0, 0.4, 0.65],[3.0, 1.0, -1.0, 0.55, 0.714]])
    # MdSimulation().start()
