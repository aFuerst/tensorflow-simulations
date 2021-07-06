import threading
from socket import *
import pickle
import zmq
import time

class Server(object):
    def __init__(self, port, host):
        # self.running = True
        self.context = zmq.Context()
        self.p1 = "tcp://" + str(host) + ":" + str(port)
        self.p2 = "tcp://" + str(host) + ":" + str(12001)
        self.s = self.context.socket(zmq.REP)
        self.sendValue = ''
        self.finalDict = {}
        self.intermediateDict = {}
        self.lock = threading.Lock()
        return


    def writedata(self, receivedpacket, dict):
        #TODO: lock should only be on the dict
        f= open("mapreduce_dictionary.log", "w")
        self.lock.acquire()
        for key in receivedpacket.keys():
            if key in dict.keys():
                dict[key].append(receivedpacket[key])
            else:
                dict[key] = [receivedpacket[key]]

        print("DEBUG :: Data written to KV store")
        self.lock.release()
        self.sendValue = 1
        self.s.send(pickle.dumps(self.sendValue))
        # pickle.dump(dict,f)
        f.write("\n"+str(dict))
        # self.context.destroy()
        return 1

    def connectclient(self):
        while 1:
           print("Server Listening .. ")
           self.context = zmq.Context()
           self.s = self.context.socket(zmq.REP)
           self.s.bind(self.p1)
           self.s.bind(self.p2)
           self.sendValue = ''
           receivedpacket = self.s.recv()

           msg_body, msg_header = pickle.loads(receivedpacket)
           print("message:",msg_header)

           # GET fetches from the dictionary
           if not "STOP" in msg_header:
               if 'GET' in msg_header:
                   if len(msg_body) > 1:
                       hashkey = msg_body[0]
                       number_reducers = msg_body[1]
                       print("fetching based on hashkey")
                       # Fetch all the values with received hashkey from intermediate dict
                       self.sendValue = self.fetch_hashkeys(hashkey, number_reducers)
                       print("\n DEBUG :: returning to reducer:")
                       self.s.send(pickle.dumps(self.sendValue))
                   # self.context.destroy()

               # STORE intermediate result to the dictionary
               elif 'INTERMEDIATE' in msg_header:
                   t = threading.Thread(target=self.writedata, args=(msg_body, self.intermediateDict))
                   t.start()

               # STORE final result to the dictionary
               elif 'FINAL' in msg_header:
                   t = threading.Thread(target=self.writedata, args=(msg_body, self.finalDict))
                   t.start()
           else:
               print("\n Stopping Server ..")

               self.context.destroy()
               break
           time.sleep(5)
           self.context.destroy()
        return

    def closeconnection(self):
        # print("\n Final Dictionary in the KV store is:", self.finalDict)
        # print("\n Stopping Server ..")
        # self.running = False
        pass


    def fetch_hashkeys(self, hashkey, number_reducers):
        res_dict = {}
        for key in self.intermediateDict.keys():
            if hash(key)%number_reducers == hashkey:
                if key in res_dict:
                    res_dict[key].append(self.intermediateDict[key])
                else:
                    res_dict[key] = self.intermediateDict[key]
        return res_dict