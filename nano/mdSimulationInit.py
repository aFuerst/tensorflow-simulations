# import numpy as np
# import argparse, math, os, datetime
# import logging
# import utility, control, interface, bin, thermostat, md, velocities, forces, common
# import tensorflow_manip as tfmanip
#
# np.random.seed(0)  # be consistent
#
#
#
# def setup_params(data):
#     cpu = True
#     verbose =
#     parser.add_argument('-c', "--cpu", action="store_true")
#     parser.add_argument('-v', "--verbose", action="store_true")
#     parser.add_argument('-x', "--xla", action="store_true")
#     parser.add_argument('-r', "--prof", action="store_true")
#     parser.add_argument('-o', "--opt", action="store_true")
#     parser.add_argument('-d', "--charge-density", action="store", default=-0.0, type=float)
#     parser.add_argument("--ein", action="store", default=80, type=float)
#     parser.add_argument("--eout", action="store", default=80, type=float)
#     parser.add_argument('-ec', "--extracompute", action="store", default=10000, type=int)
#     parser.add_argument('-mf', "--moviefreq", action="store", default=10000, type=int)
#     parser.add_argument('-he', "--hiteqm", action="store", default=100000, type=int)
#     parser.add_argument('-t', "--delta-t", action="store", default=0.001, type=float)
#     parser.add_argument('-s', "--steps", action="store", default=100, type=int)
#     parser.add_argument('-f', "--freq", action="store", default=10, type=int)
#     parser.add_argument('-cf', "--cppfreq", action="store", default=100, type=int)
#     parser.add_argument('-wd', "--writedensity", action="store", default=100000, type=int)
#     parser.add_argument('-th', "--threads", action="store", default=os.cpu_count(), type=int)
#     parser.add_argument("--validate", action="store_true")
#     parser.add_argument("--random-pos-init", action="store_false")
#     parser.add_argument('-bw', "--bin_width", action="store", default=0.05,
#                         type=float)  # this is in reduced units to fix it
#     parser.add_argument('-fd', "--fraction_diameter", action="store", default=1 / 28.0, type=float)
#     parser.add_argument('-chl', "--chain_length_real", action="store", default=5, type=float)
#     parser.add_argument('-Q', "--therm_mass", action="store", default=1, type=float)
#     parser.add_argument('--bx', action="store", default=15.3153, type=float)
#     parser.add_argument('--by', action="store", default=15.3153, type=float)
#     parser.add_argument('-M', "--concentration", action="store", default=self.conc, type=float)
#     parser.add_argument('-e', "--pos-valency", action="store", default=self.pos, type=int)  # changed from 1 to 0
#     parser.add_argument('-en', "--neg-valency", action="store", default=self.neg, type=int)  # changed from -1 to 0
#     parser.add_argument('-cl', "--confinment-len", action="store", default=self.conf_len, type=float)
#     parser.add_argument('-pd', "--pos-diameter", action="store", default=self.diam, type=float)
#     parser.add_argument('-nd', "--neg-diameter", action="store", default=self.diam, type=float)
#     logging.basicConfig(filename="logs_" + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + ".log",
#                         format='%(asctime)s %(message)s',
#                         filemode='w')
#     logger = logging.getLogger()
#     logger.setLevel(logging.DEBUG)
#     tfmanip.toggle_xla(args.xla)
#     tfmanip.manual_optimizer(args.opt)
#     config = tfmanip.toggle_cpu(args.cpu, args.threads)
#     tfmanip.silence()
#     self.start_sim(config, args, logger)
#
#
# if __name__ == "__main__":
#     MdSimulation().start()
#     MdSimulation().start()