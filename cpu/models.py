import numpy as np
import sys, os

dir_name = os.path.dirname(__file__)
sys.path.append(dir_name)

from functions import *
from struct import unpack     


class DARUMA:
    def __init__(self):
        self.feature = load_AAindex()
        self.model = CNN3_128_9_NN2_121_128()

    def predict_from_seqence(self, seq, threshold, smoothing_window=17, remove_short_regions=True):
        
        x = np.array([self.feature[res] for res in seq],dtype="float32")
        prob = self.model(x)[:,1].tolist()

        if smoothing_window:
            prob = smoothing(smoothing_window,prob)
        
        pred = classify(prob,threshold=threshold)

        if remove_short_regions:
            pred = remove_short_idr(pred)
            pred = remove_short_stru(pred)

        return prob,pred
    
    

class CNN3_128_9_NN2_121_128:
    def __init__(self):
        self.params = {}

        weight_parameters_path = os.path.join(dir_name, 'data', 'CNN3_128_9_NN2_121_128.weight')
        with open(weight_parameters_path,"rb") as f:
            binary_data = f.read()
        data = unpack(">2931714f",binary_data)

        self.params["W1"] = np.array(data[0:637056],dtype="float32").reshape(9, 553, 128)
        self.params["b1"] = np.array(data[637056:637184],dtype="float32")
        self.params["W2"] = np.array(data[637184:784640],dtype="float32").reshape(9, 128, 128)
        self.params["b2"] = np.array(data[784640:784768],dtype="float32")
        self.params["W3"] = np.array(data[784768:932224],dtype="float32").reshape(9, 128, 128)
        self.params["b3"] = np.array(data[932224:932352],dtype="float32")

        self.params["W4"] = np.array(data[932352:2914816],dtype="float32").reshape(121,128,128)
        self.params["b4"] = np.array(data[2914816:2914944],dtype="float32")
        self.params["W5"] = np.array(data[2914944:2931328],dtype="float32").reshape(128,128)
        self.params["b5"] = np.array(data[2931328:2931456],dtype="float32")
        self.params["W6"] = np.array(data[2931456:2931712],dtype="float32").reshape(128,2)
        self.params["b6"] = np.array(data[2931712:2931714],dtype="float32")

        self.conv1 = D1Conv(self.params["W1"],self.params["b1"],active=relu)
        self.conv2 = D1Conv(self.params["W2"],self.params["b2"],active=relu)
        self.conv3 = D1Conv(self.params["W3"],self.params["b3"],active=relu)
        self.conv4 = D1Conv(self.params["W4"],self.params["b4"],active=relu)
        self.affine1 = Affine(self.params["W5"],self.params["b5"],active=relu)
        self.affine2 = Affine(self.params["W6"],self.params["b6"],active=softmax)

        self.layers = [self.conv1,self.conv2,self.conv3,self.conv4,self.affine1,self.affine2]

    def __call__(self,x):
        x = np.pad(x, [(72,72),(0,0)], 'constant')
        for layer in self.layers:
            x = layer.forward(x)

        return x