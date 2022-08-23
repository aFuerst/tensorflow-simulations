# Master is a library consisting of one class called Cluster, for now. It will initiate a cluster whenever its object is created.
# The cluster will have properties like number of mappers, number of reducers to start a cluster.
# It will have other operations (like run_mapred, destroy_cluster etc.) in the form of functions calls.

import dill
import os, signal
import time
import configparser
# import server
import importlib
import data_generator, nano_init, surrogate

class Cluster:
    def __init__(self):
        self.training_complete = False
        self.threshold_loss = 0.0001
        self.serverport = 0
        self.serverhost = 0
        self.ready_to_predict = False
        self.filenames = []
        self.surrogate_function = ''
        self.tfmd_function = ''
        self.train_volume = 0
        self.datagenerator_obj = data_generator.Data_generator()
        self.surrogate_obj = surrogate.Surrogate()
        self.tfmd_obj = nano_init.MdSimulation()
        self.density_profile_file = ''
        # self.parser = configparser.ConfigParser()
        self.config_cluster()
        return

    def config_cluster(self):
        self.config = configparser.ConfigParser()
        self.config.read('config.ini')
        self.serverport = self.config.getint('DEFAULT', 'server_port')
        self.hostname = self.config['DEFAULT']['host']
        self.surrogate_function = self.config['DEFAULT']['surrogate_function']
        self.surrogateclass = self.config['DEFAULT']['surrogate_class']
        self.tfmd_function = self.config['DEFAULT']['tfmd_function']
        self.tfmdclass = self.config['DEFAULT']['tfmd_class']
        self.density_profile_file = self.config['DEFAULT']['density_profile']
        self.input_file = self.config['DEFAULT']['input_file']
        return

    def run_simulation(self):
        print("Checking if Model is trained:", self.training_complete)
        # if not self.training_complete:
        average_loss = 1.0
        while not self.training_complete:
            data_chunk = self.datagenerator_obj.generate_bulk_input()

            # Starting MD Simulation on some systems to generate density profiles
            self.tfmd_obj.start(data_chunk[0:4]) #, self.input_file)

            # Train the surrogate on generated density profiles
            average_loss = self.surrogate_obj.train_model()
            print("Average loss:", average_loss, "threshold_loss:", self.threshold_loss)

            # Check if more training is needed
            if average_loss <= self.threshold_loss:
                self.training_complete = True
                print("Training completion ::", self.training_complete)
                self.config.set('DEFAULT', 'training_complete', 'True')
                with open('config.ini', 'w') as configfile:
                    self.config.write(configfile)

        # else:
        prediction_res = self.surrogate_obj.predict(lis=[3.8, 3, -1, 0.55, 0.700], filename=self.density_profile_file)
        return prediction_res

    def start_tfmd(self):
        nano_init.start(self.serverport, self.serverhost)


    def start_surrogate(self):
        # components = self.surrogateclass.split('.')
        # module = __import__(components[0])
        # for cls in components[1:]:
        #     module = getattr(module, cls)
        # self_inst = module()
        # module = getattr(module, self.surrogate_function)
        # self.surrogate_function = dill.dumps(module)
        # print("module::", module)
        # surrogate_pid = os.fork()
        # if surrogate_pid == 0:
        #     surrogate_func = dill.loads(self.surrogate_function)
        #     average_loss = surrogate_func(self_inst, self.serverport, self.serverhost)
        # else:
        #     # Main thread does nothing else for now
        #     pass
        average_loss = self.surrogate_obj.run_surrogate(self.serverport, self.serverhost)


    def destroy_cluster(self):
        # os.killpg(os.getpid(), signal.SIGTERM)
        pass

    # [DEFAULT]
    # server_port = 12000
    # host = 127.0
    # .0
    # .1
    # surrogate_function = run_surrogate
    # surrogate_class = surrogate.Surrogate
    # tfmd_function = start
    # tfmd_class = nano_init
    # density_profile = output / predict / density_profile.dat
    # training_complete = False
    # threshold_loss = 0.5


