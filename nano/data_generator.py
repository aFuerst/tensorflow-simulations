import os
import random
import pickle
import pandas as pd

'''This class would generate all possible nano confinement systems configuration using different combinations of confinement length, positive valency, neg valency, concentration, pos and neg diameters. These nano conf systems will be used by MDSimulation code to generate density profiles and those dnsity profiles will be used by the surrogate for training'''
class Data_generator:
    def __init__(self):
        self.file_all_data = "input/all_data.dat"
        self.file_used_data = "input/data_dictionary.txt"
        self.dict_used_indexes = {}
        self.chunk = 10
        return

    def generate_bulk_input(self):
        # print("Generating input system combinations ...")
        if not os.path.exists(self.file_all_data):
            return -1
        if not os.path.exists(self.file_used_data):
            return -1

        with open(self.file_used_data, 'rb') as file_obj:
            try:
                mydict = pickle.load(file_obj)
            except EOFError:
                mydict = {}

        # print("Initially reading:", len(mydict), ":", mydict.keys())
        df = pd.read_csv(self.file_all_data, header=None, sep="\t")
        randlist = []
        while len(randlist) < self.chunk and len(mydict.keys()) < len(df):
            idx = random.randint(0, len(df)-1)
            if idx in mydict.keys():
                continue
            else:
                randlist.append(df.iloc[idx].tolist())
                mydict[idx] = df.iloc[idx]

        # print("after generating:", len(mydict))
        with open(self.file_used_data, 'wb') as file_obj:
            pickle.dump(mydict, file_obj)
        print(mydict.keys())
        return randlist


        # records = 0

        # TODO : create a bag of jobs
        # z_val = [3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4.0]
        # p_val = [1, 2, 3]
        # n_val = [-1, - 2]
        # c_val = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
        # d_val = [0.5, 0.55, 0.553, 0.6, 0.65, 0.7, 0.714, 0.75]
        # while records < 100:
        #     z = random.sample(z_val, 1)[0]
        #     p = random.sample(p_val, 1)[0]
        #     n = random.sample(n_val, 1)[0]
        #     c = random.sample(c_val, 1)[0]
        #     d = random.sample(d_val, 1)[0]
        #     # print(z, p)
        #     file_obj.write(str(z) + "\t" + str(p) + "\t" + str(n) + "\t" + str(c) + "\t" + str(d)+ "\n")
        #     records += 1
        # print("Input system combinations generated and written to file:", self.bulk_file)
        # file_obj.close()

        # for z_val in [3.0, 4.0, 3.5, 3.2, 3.7, 3.1, 3.3, 3.6, 3.8, 3.4, 3.9]:   #, [3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4.0]:
        #     for p_val in [1, 2, 3]:
        #         for n_val in [-1, - 2]:
        #             for c_val in [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]:
        #                 for d_val in [0.5, 0.55, 0.553, 0.6, 0.65, 0.7, 0.714, 0.75]:
        #                     # save in a file here
        #                     # file_obj.write(z_val+"\t"+p_val+"\t"+n_val+"\t"+c_val+"\t"+d_val)
        #                     records += 1



obj = Data_generator()
data = obj.generate_bulk_input()
print(data)