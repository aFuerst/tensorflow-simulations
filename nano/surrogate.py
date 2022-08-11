# This file consists of the the surrogate that trains on data produced by tensorflow MD simulations
import joblib
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.layers import Dense
from tensorflow.keras.layers import Dropout
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow_core.python.keras.optimizers import SGD
import matplotlib.pyplot as plt
import random
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle

print(keras.__version__)
print(tf.__version__)

class Surrogate:
    def __init__(self):
        self.port = None
        self.host = None
        self.filename = 'output/collectDatawhole.dat' #'input/p_density_profile.dat'    #nano.utility.root_path
        self.checkpoint_path = "output/training/cp.ckpt"
        self.train_errorBars = None
        self.test_errorBars = None
        self.train_output = None
        self.test_output = None
        self.train_input = None
        self.train_pos = None
        self.test_input = None
        self.model = None    # load the existing model if any

        self.test_pos = None
        self.history = None
        self.file_success_rate = "output/training/success_rate"
        self.file_loss = "output/training/loss"

        self.beta_1 = 0.9
        self.beta_2 = 0.999
        self.decay = 0.000000
        self.input_features = 5
        self.hidden_layer1 = 512
        self.hidden_layer2 = 256
        self.output_neurons = 150
        self.learning_rate = 0.0001  # 0.9
        self.epochs = 400
        self.dropout_rate = 0.15
        self.sgd = SGD(lr=0.01, momentum=0.9)
        self.batch_size = 15
        self.train_test_split = 0.8


    def load_data(self, df):
        df.head()
        x = df.loc[:, '#Z_val':'d_val']            #df.iloc[:, 0:5]  # df[0:][0:5]
        y = df.loc[:, 'stringPosition0':]          #df.iloc[:, 5:]
        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = self.train_test_split)

        # Training Data
        train_pos = y_train.iloc[:, 0:150]
        train_density = y_train.iloc[:, 150:300]
        train_errorBars = y_train.iloc[:, 300:450]

        # Testing Data
        test_pos = y_test.iloc[:, 0:150]
        test_density = y_test.iloc[:, 150:300]
        test_errorBars = y_test.iloc[:, 300:450]

        # Scale the input data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_x_train = scaler.fit_transform(x_train)
        scaled_x_test = scaler.transform(x_test)

        # Save the scaler for future use
        joblib.dump(scaler, 'output/training/scaler_new.pkl')

        # initializing class attributes
        self.train_input = scaled_x_train
        self.train_errorBars = train_errorBars
        self.train_output = train_density
        self.train_pos = train_pos
        self.test_input = scaled_x_test
        self.test_errorBars = test_errorBars
        self.test_output = test_density
        self.test_pos = test_pos
        print("**** Data loaded successfully ****")


    def create_model(self):
        initializer = tf.keras.initializers.VarianceScaling(scale=2.0, mode='fan_in', distribution='truncated_normal',
                                                            seed=None)
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(self.hidden_layer1, activation=tf.nn.relu, kernel_initializer=initializer,
                                        input_shape=(self.input_features,)))
        model.add(tf.keras.layers.Dropout(rate=self.dropout_rate))
        model.add(tf.keras.layers.Dense(self.hidden_layer2, activation=tf.nn.sigmoid, kernel_initializer=initializer))
        model.add(tf.keras.layers.Dropout(rate=self.dropout_rate))
        model.add(tf.keras.layers.Dense(self.output_neurons, kernel_initializer=initializer))
        model.compile(loss=tf.keras.losses.mean_squared_error,
                      optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate, beta_1=self.beta_1, beta_2=self.beta_2, decay=self.decay))


        # model = keras.Sequential()
        # model.add(Dense(self.hidden_layer1, input_dim=self.input_features, kernel_initializer=tf.keras.initializers.Ones(), activation='relu',
        #                 name='input_layer'))
        # model.add(Dropout(self.dropout_rate))
        # model.add(Dense(self.hidden_layer2, activation='relu', name="hidden_layer"))
        # model.add(Dropout(self.dropout_rate))
        # model.add(Dense(self.output_neurons, activation='linear', name="output_layer"))
        # model.compile(optimizer='adam', loss="mean_squared_error", metrics=['accuracy'])

        # model.summary()
        return model

    def train_model(self, train_data):
        self.load_data(train_data)
        model = self.create_model()
        # if os.path.isfile(self.checkpoint_path) :
        #     model.load_weights(self.checkpoint_path)
        # checkpoint_dir = os.path.dirname(self.checkpoint_path)
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=self.checkpoint_path,
                                                         save_weights_only=True) #verbose=1
        # Train the model with the new callback
        self.history = model.fit(self.train_input, self.train_output, epochs=self.epochs, batch_size=self.batch_size, verbose=1,
                            validation_data=(self.test_input, self.test_output), shuffle=True, callbacks=[cp_callback])
        scaler = joblib.load('output/training/scaler_new.pkl')
        predictions = model.predict(scaler.transform(self.test_input))
        success_rate = self.calc_success_rate(predictions)
        return success_rate

    def calc_success_rate(self, predictions):
        accuracy = []
        print("predictions.shape[0], predictions.shape[1]:", predictions.shape[0], predictions.shape[1])
        for i in range(predictions.shape[1]):
            row_hits = 0
            for j in range(predictions.shape[0]):
                if abs(predictions[j, i] - self.test_output.iloc[j][i]) <= self.test_errorBars.iloc[j][i]:
                    row_hits = row_hits + 1
            accuracy.append(row_hits / predictions.shape[0])
        return np.array(accuracy)

    def predict(self, lis):
        model = self.create_model()
        # if not os.path.isdir(self.checkpoint_path) :
        #     print("Model checkpoints not found. Returning blank prediction")
        #     return
        model.load_weights(self.checkpoint_path)
        # z_vals = []
        # d = -lis[0]/2
        # while d<=0:
        #     z_vals.append(d)
        #     d += lis[0]/300
        scaler = joblib.load('output/training/scaler_new.pkl')
        file_obj = open("output/predict/density_profile.dat", 'w')
        test_input = (np.array([lis,]))
        test_input.reshape((5,))
        prediction = model.predict(scaler.transform(test_input))
        for i in range(len(prediction[0])-1, 0, -1):
            file_obj.write(str(self.train_pos.iloc[0][i])+ "\t"+ str(prediction[0][i])+"\n")
        file_obj.close()
        return prediction

    def plot_loss(self, history, c):
        plt.close()
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.yscale('log')
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper right')
        # plt.show()
        plt.savefig('output/plots/modelloss_'+ str(c) + '.png')

    def plot_density_profile(self, ground_truth, predictions, counter):
        plt.close()
        plt.plot(ground_truth[5:155], predictions[0], 'g+')
        # plt.errorbar(ground_truth[5:155], predictions[0], yerr=ground_truth[305:], fmt=" ")
        plt.plot(ground_truth[5:155], ground_truth[155:305], 'ro')
        plt.title('density profile')
        plt.ylabel('density')
        plt.xlabel('epoch')
        plt.legend(['prediction', 'ground truth'], loc='upper right')
        plt.savefig('output/plots/densityprediction_' + str(counter) + '.png')
        return

    def plot_success_rate(self, success_rate, counter):
        plt.close()
        plt.plot(success_rate, 'go--')
        plt.savefig('output/plots/success_rate_' + str(counter) + '.png')
        return

    def run_surrogate(self, port, host):
        self.port = port
        self.host = host
        df = pd.read_csv(self.filename, sep=",")
        print("total records read:", len(df.iloc[0]))
        test_row = random.randint(0, len(df)-1)
        # unseen_data = df.iloc[test_row]
        i=0
        avg_success_rate = []
        curr_loss = 1.0
        counter = 1
        while i < len(df) and curr_loss > 0.00001:
            di = min(len(df), 1000)
            succ_rate_arr = self.train_model(df.iloc[0 : i+di])
            # succ_rate_arr = self.train_model(df)
            avg_success_rate.append(sum(succ_rate_arr)/len(succ_rate_arr))
            curr_loss = self.history.history['loss'][-1]
            predictions = self.predict(df.iloc[test_row][0:5])
            self.plot_success_rate(avg_success_rate, counter)
            self.plot_loss(self.history, counter)
            self.plot_density_profile(df.iloc[test_row], predictions, counter)
            counter += 1
            i += di


        # predicting for test data
        # print("Starting prediction on user input now ::")
        # self.predict(lis=[3.0, 1, -1, 0.75, 0.5])
        return avg_success_rate[-1]


if __name__ == "__main__":
    # df = pd.read_table('input/p_density_profile.dat', sep=",")
    # print("total records read:", len(df))
    # print(df.iloc[0])
    # print(df.iloc[1])
    Surrogate().run_surrogate('','')







# print("loading training data in chunks of 1000 records ::")
        # records = 1
        # df = pd.read_table(self.filename,
        #                    sep=",",
        #                    skiprows=records, nrows=1000)
        # records += 1000
        # df.head()
        # avg_loss = 1
        # while len(df) and avg_loss > 0.001:
        #     avg_loss = self.train_model(df)
        #     if len(df) < 999:
        #         break
        #     df = pd.read_table(self.filename, sep=",", skiprows=records, nrows=1000)
        #     print(records, avg_loss, len(df))
        #     records += 1000
        #     break
        # self.predict(lis=[3.4, 1, -1, 0.75, 0.5])
