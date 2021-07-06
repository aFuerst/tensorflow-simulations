# This file consists of the the surrogate that trains on data produced by tensorflow MD simulations
# %tensorflow_version 2.x
#import libraries and check versions
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.keras import initializers
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
from tensorflow_core.python.keras.optimizers import SGD

print(keras.__version__)
print(tf.__version__)

class Surrogate:
    def __init__(self):
        self.port = None
        self.host = None
        self.filename = 'output/collectDatawhole.dat'  #nano.utility.root_path
        self.train_errorBars = None
        self.test_errorBars = None
        self.train_output = None
        self.test_output = None
        self.train_input = None
        self.test_input = None
        self.model = None
        self.learning_rate = 0.0001
        self.epochs = 7000
        self.dropout_rate = 0.15
        self.sgd = SGD(lr=0.01, momentum=0.9)
        self.accuracy = 0

    '''
    This functions loads the training data from shared path. 
    Input : None
    Output: train and test data
    '''
    def load_data(self):
        df = pd.read_table(self.filename,
                           sep=",",
                           skiprows=1)
        df.head()
        x = df.iloc[:, 0:5]  # df[0:][0:5]
        y = df.iloc[:, 5:]
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
        train_density = y_train.iloc[:, 150:300]
        train_errorBars = y_train.iloc[:, 300:450]
        train_pos = y_train.iloc[:, 0:150]
        test_errorBars = y_test.iloc[:, 300:450]
        test_density = y_test.iloc[:, 150:300]
        test_pos = y_test.iloc[:, 0:150]

        # clean-up densities to remove zero values
        train_density = train_density.iloc[:, 18:]
        train_errorBars = train_errorBars.iloc[:, 18:]
        train_pos = train_pos.iloc[:, 18:]
        test_errorBars = test_errorBars.iloc[:, 18:]
        test_density = test_density.iloc[:, 18:]
        test_pos = test_pos.iloc[:, 18:]

        print(x_train.shape, train_density.shape, train_errorBars.shape, train_pos.shape)

        # initializing class attributes
        self.train_input = x_train
        self.train_errorBars = train_errorBars
        self.train_output = train_density
        self.test_input = x_test
        self.test_errorBars = test_errorBars
        self.test_output = test_density

    def initialize_model(self):
        # self.model = keras.models.Sequential([
        #     keras.layers.Dense(512, input_shape=(5,), kernel_initializer=keras.initializers.glorot_normal(seed=None),
        #                           kernel_constraint=maxnorm(3), activation="relu"),
        #     keras.layers.Dropout(self.dropout_rate),
        #     keras.layers.Dense(256, kernel_initializer=keras.initializers.glorot_normal(seed=None),
        #                           kernel_constraint=maxnorm(3), activation="relu"),
        #     keras.layers.Dropout(self.dropout_rate),
        #     keras.layers.Dense(134, kernel_initializer=keras.initializers.glorot_normal(seed=None),
        #                           activation="linear")

        self.model = keras.Sequential()
        self.model.add(Dense(512, input_dim=5, kernel_initializer=tf.keras.initializers.Ones(), activation='relu',
                        name='input_layer'))
        self.model.add(Dropout(self.dropout_rate))
        self.model.add(Dense(256, activation='relu', name="hidden_layer"))
        self.model.add(Dropout(self.dropout_rate))
        self.model.add(Dense(132, activation='linear', name="output_layer"))
        opt = keras.optimizers.Adam(learning_rate=0.001)
        self.model.summary()


    def train_model(self):
        # self.epochs = 10
        self.model.compile(optimizer='adam', loss="mean_squared_error", metrics=['accuracy'])
        self.model.fit([self.train_input], [self.train_output], epochs=self.epochs, batch_size=5, validation_split=0.2)

    def predict(self):
        prediction = self.model.predict([self.test_input])
        # predicted_peaks = []
        # gt_peaks = []
        # for i, ele in enumerate(prediction):
        #     ind = list(ele).index(max(ele))
        #     predicted_peaks.append(ele[ind])
        #     gt_peaks.append(self.test_output.iloc[i][ind])
        # print(list(prediction[0]))
        # plt.plot(self.test_pos[0], list(prediction[0]), "b^")
        # plt.plot(self.test_pos[0], self.test_output[0], "r*")

        # Calculate performance
        predicted_data = pd.DataFrame(prediction)
        hits = 0
        for indices in range(len(predicted_data)):
            row_hits = 0
            l = len(predicted_data.iloc[indices])
            for j in range(l):
                lower = self.test_output.iloc[indices][j] - self.test_errorBars.iloc[indices][j]
                upper = self.test_output.iloc[indices][j] + self.test_errorBars.iloc[indices][j]
                if predicted_data.at[indices, j] <= upper and predicted_data.at[indices, j] >= lower:
                    row_hits += 1
            if row_hits > l / 4:
                hits += 1

        acc = hits / (indices - 1)
        print("Accuracy:", acc * 100, "% hits:", hits, " ", len(predicted_data))

    def clean_data(self):
        pass

    def run_surrogate(self, port, host):
        self.port = port
        self.host = host
        print("loading data ===>")
        self.load_data()
        print("initializing model ===>")
        self.initialize_model()
        print("training model ===>")
        self.train_model()
        print("predicting model ===>")
        self.predict()
        return self.accuracy


# if __name__ == "__main__":
#     Surrogate.main(Surrogate())