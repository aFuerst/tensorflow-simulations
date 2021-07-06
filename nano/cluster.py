# Master is a library consisting of one class called Cluster, for now. It will initiate a cluster whenever its object is created.
# The cluster will have properties like number of mappers, number of reducers to start a cluster.
# It will have other operations (like run_mapred, destroy_cluster etc.) in the form of functions calls.

import dill
import os, signal
import time
import configparser
import server
import importlib

class Cluster:
    def __init__(self):
        self.ready_to_train = False
        self.acceptable_loss = 0
        self.serverport = 0
        self.serverhost = 0
        self.ready_to_predict = False
        self.filenames = []
        self.surrogate_function = ''
        self.tfmd_function = ''
        self.train_volume = 0

        return

    def config_cluster(self):
        config = configparser.ConfigParser()
        config.read('config.ini')
        self.serverport = config.getint('DEFAULT', 'server_port')
        self.hostname = config['DEFAULT']['host']
        self.surrogate_function = config['DEFAULT']['surrogate_function']
        self.surrogateclass = config['DEFAULT']['surrogate_class']
        self.tfmd_function = config['DEFAULT']['tfmd_function']
        self.tfmdclass = config['DEFAULT']['tfmd_class']
        return

    def run_input_datagenerator(self):
        while not self.ready_to_train :
            res = self.generate_bulk_input()
            # keeping it simple for now. This can be changed to incremental data generation later on
            self.ready_to_train = res


    def generate_bulk_input(self):
        for Z_val in [3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4.0]:
            for p_val in [1, 2, 3]:
                for n_val in [-1, - 2]:
                    for c_val in [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]:
                        for d_val in [0.5, 0.55, 0.553, 0.6, 0.65, 0.7, 0.714, 0.75]:
                            # save in a variable here
                            pass

    def run_tfmd(self):
        self.tfmd_function = dill.dumps(getattr(__import__(self.tfmdclass), self.tfmd_function))
        tfmd_pid = os.fork()
        if tfmd_pid == 0:
            run_tfmd = dill.loads(self.tfmd_function)
            input_data_size = run_tfmd(self.serverport, self.serverhost)
            if input_data_size > self.train_volume:
                self.ready_to_train = True
            else:
                # TODO: generate more input data
                pass
        else:
            # Main thread does nothing else for now
            pass

    def start_surrogate(self):
        self.surrogate_function = dill.dumps(getattr(__import__(self.surrogateclass), self.surrogate_function))
        surrogate_pid = os.fork()
        if surrogate_pid == 0:
            surrogate_func = dill.loads(self.surrogate_function)
            average_loss = surrogate_func(self.serverport, self.serverhost)
            if average_loss <= self.acceptable_loss:
                self.ready_to_predict = True
            else:
                # TODO: generate more input data
                pass
        else:
            # Main thread does nothing else for now
            pass

    def destroy_cluster(self):
        # os.killpg(os.getpid(), signal.SIGTERM)
        pass