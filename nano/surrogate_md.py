# This file consists of the the surrogate that trains on data produced by tensorflow MD simulations

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.optimizers import SGD
from keras.constraints import maxnorm
from sklearn.model_selection import StratifiedKFold
import nano.utility

class Surrogate:
    def __init__(self):
        self.filename = nano.utility.root_path
        self.train_pos = None
        self.test_pos = None
        self.train_density = None
        self.test_density = None
        self.train_input_params = None
        self.test_input_params = None
        self.model = None
        self.learning_rate = 0.0001
        self.epochs = 7000
        self.dropout_rate = 0.15
        self.sgd = SGD(lr=0.01, momentum=0.9)

    '''
    This functions loads the training data from shared path. 
    Input : None
    Output: train and test data
    '''
    def load_data(self):
        contents = open(self.filename).read().splitlines()
        contents = contents[1:]
        # random.shuffle(contents)
        input_params = []
        output_density = []
        pos_val = []
        error_bars = []
        num_rows = len(contents)
        for line in range(1, num_rows - 1):
            line_arr = contents[line].split(",")
            input_params.append(list(map(float, line_arr[0:5])))
            pos_val.append(list(map(float, line_arr[21:155])))
            output_density.append(list(map(float, line_arr[171:305])))
        train_size = 3 * num_rows // 5
        train_input, test_input = input_params[0:train_size], input_params[train_size:]
        train_pos, test_pos = pos_val[0:train_size], pos_val[train_size:]
        train_gt, test_gt = output_density[0:train_size], output_density[train_size:]
        print(len(test_gt), len(test_pos))

    def initialize_model(self):
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(512, input_shape=(5,), kernel_initializer=keras.initializers.glorot_normal(seed=None),
                                  kernel_constraint=maxnorm(3), activation="relu"),
            tf.keras.layers.Dropout(self.dropout_rate),
            tf.keras.layers.Dense(256, kernel_initializer=keras.initializers.glorot_normal(seed=None),
                                  kernel_constraint=maxnorm(3), activation="relu"),
            tf.keras.layers.Dropout(self.dropout_rate),
            tf.keras.layers.Dense(134, kernel_initializer=keras.initializers.glorot_normal(seed=None),
                                  activation="linear")
        ])

    def train_model(self):
        self.model.fit(self.train_input, self.train_density, epochs=self.epochs)

    def predict(self):
        prediction = self.model.predict(self.test_input)
        predicted_peaks = []
        predicted_pos_peaks = []
        gt_peaks = []
        for i, ele in enumerate(prediction):
            ind = list(ele).index(max(ele))
            predicted_peaks.append(ele[ind])
            # predicted_pos_peaks.append(pos_val[i][ind])
            gt_peaks.append(self.test_density[i][ind])
        print(list(prediction[0]))
        plt.plot(self.test_pos[0], list(prediction[0]), "b^")
        plt.plot(self.test_pos[0], self.test_density[0], "r*")

    def clean_data(self):
        pass