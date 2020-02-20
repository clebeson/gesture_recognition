
import numpy as np
import csv
import sys
import os
import argparse
import math
import glob
import pickle
import random
import json
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable
import bayesian_layers as bl
import random 
from skeleton import *
from spotting import *
import matplotlib.pyplot as plt
torch.manual_seed(30)
np.random.seed(30)
random.seed(30) 



class GestureRecognizer(nn.Module):
    
    def __init__(self, model_path, dropout = 0.1, logstd1 = -1, logstd2 = -2, pi = 0.5 ):
        super(GestureRecognizer, self).__init__()
        type =  bl.ModelType.MC_DROP
        
        self.window = []
        self.movement = False
        self.window_size = 3
        self.spotting = GestureSpotting()
        self.spotting.load("model_spotting3.pth")
        self._baysian_layers = []

        linear_args =  {
            "mu":0,
            "logstd1":logstd1,
            "logstd2":logstd2,
            "pi":pi,
            "type":type,
            "dropout":dropout}

        rnn_args =  {
            "mu":0,
            "logstd1":logstd1,
            "logstd2":logstd2,
            "pi":pi,
            "type":type,
            "dropout":dropout}

        last_linear_args = {
            "mu":0,
            "logstd1":logstd1,
            "logstd2":logstd2,
            "pi":pi,
            "type":type,
            "dropout":0}
    
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    
        model = torch.load(model_path,  map_location=self.device)
        self.input_size = 21
        self.embeding_size = 16
        self.output_size = 15
        self.hidden_dim = model["hidden_dim"]
        self.n_layers = model["num_layers"]
 
        self.bfc1 = bl.Linear(self.input_size, 32,**linear_args) #128
        self.bfc2 = bl.Linear(32, self.embeding_size, **linear_args)
        self.fc1 = nn.Sequential(self.bfc1,nn.ReLU())
        self.fc2 = nn.Sequential(self.bfc2,nn.ReLU())
        self.lstm = bl.LSTM( input_size = self.embeding_size,\
                                    hidden_size = self.hidden_dim, num_layers = self.n_layers, \
                                    batch_first=True,**rnn_args)  
                                    


        # dropout layer
        self.fc_combine = bl.Linear(self.input_size, self.input_size, **linear_args)
        self.combine = nn.Sequential(self.fc_combine, nn.ReLU())
        
        # Classifier layer
        self.fc = bl.Linear(self.hidden_dim, self.output_size, **last_linear_args)
        
        self._load(model_path)
    
 
    def _load(self,path):
        self = super().to(self.device) 
        self.fc1 = self.fc1.to(self.device) 
        self.fc2 = self.fc2.to(self.device)
        self.lstm = self.lstm.to(self.device)
        self.fc = self.fc.to(self.device)

        model = torch.load(path,  map_location=self.device)["model"]
        self.load_state_dict(model, strict=False)
        self.eval()
        

    def forward(self, x, hidden):
        
        emb = self.fc1(x)
        emb = self.fc2(emb)
        
        out, hidden_out = self.lstm(emb, hidden)
        out = self.fc(out) 

        out = out.contiguous().view(-1, out.size(-1))
        return out, hidden_out
    def set_dropout(self, value):
        self.dropout = value
        for layer in self.get_baysian_layers(): layer.dropout = value

    def get_baysian_layers(self):
        if not self._baysian_layers:
            self._baysian_layers = [module for module in self.modules() if isinstance(module,bl.BaseLayer)]
        return self._baysian_layers

    def _calc_uncertainty(self,probs):
        if len(probs.shape) > 2:
            mean = probs.mean(1)
            h = -(mean*np.log(mean)).sum(1) #entropy
        else: 
            mean = probs.mean(0)
            h = -(mean*np.log(mean)).sum(0) #entropy
            # s = probs.std(0).sum()
        return h

    def _data_transformation(self, skl, mc_samples = 20):
        skl = skl.vectorize_reduced()
        data = np.expand_dims(skl,0)
        # data = (skl-self.mean)/self.std
        #creating a batch with same data. It improves the performance of MC samples
        data = np.repeat(data, mc_samples, 0)
        data = torch.from_numpy(data).float().unsqueeze(1)
        data = data.to(self.device)
        return data

    def predict(self, skeleton):
        mc_samples = 100
        self.set_dropout(0.1)
        spp_pred, spp_prob, spp_unc = self.spotting.predict(skeleton,mc_samples)
        if len(self.window) == self.window_size: del self.window[0]
        self.window.append(spp_pred)
        w = sum(self.window)/self.window_size


        if w == 1.0 and not self.movement: 
            self.movement = True
        elif w == 0.0 and self.movement: 
            self.movement = False
        
        # print(w, spp_pred, self.movement)
        
        if not self.movement:
            self.hidden_state = None
            return 0, spp_prob, spp_unc
        

        data = self._data_transformation(skeleton, mc_samples)
        with torch.no_grad():
                out,hidden = self.forward(data,self.hidden_state)
                out = out.cpu()
                # self.hidden_state = ([h.data.mean(1,keepdim=True).repeat(1,mc_samples,1) for h in hidden])
                self.hidden_state = ([h.data for h in hidden])
                probs = F.log_softmax(out,1).exp().detach().numpy()
                uncertainty = self._calc_uncertainty(probs)
                mean = probs.mean(0)
                pred = np.argmax(mean) + 1
                prob = mean.max()       
        return pred, prob, uncertainty     
    
if __name__ == "__main__":
    import time
    import pickle

    args = sys.argv
    model_path =  "../saved_models/model_gesture1_72.00.pth" if len(args)==1 else args[1]
    model = GestureRecognizer(model_path)

    # model.load(model_path)
    # ckp = torch.load("model_gesture3_0_91.16.pth")
    # for var_name in ckp:
    #     print(var_name, "\t",ckp[var_name].shape)
    
    files = glob.glob("/public/datasets/ufes-2020-01-23/*3d.json")

    #files = [ file for file in files if file[-11:-8] in ["g01","g06", "g10", "g12", "g15"]]
    print(len(files))
    results =  [[[] for _ in range(15)] for _ in range(10)]

    
    for fps in range(1,11):
        sampling_rate = 1000/fps
        
        unc = 0.0
        unc_spt = 0.0
        correct = 0.0

        total = 0.0
        total_spt = 0.0
        for file in files:
            gesture = int(file[-10:-8])
            
            with open(file) as f:
                try:
                    data = json.load(f)
                except: pass
                previous_sample = -1
                for i, localization in enumerate(data["localizations"]):
                    sample = i*100
                    if sample // sampling_rate <= previous_sample: continue
                    previous_sample = (sample // sampling_rate)
                    annotations = ObjectAnnotations(localization) 
                    skeletons = [Skeleton(obj) for obj in annotations.objects]
                    for skl in skeletons:
                        #you can insert here a method to filter the principal skeleton
                        skl_normalized = skl.normalize()
                        #skl_vector = skl_normalized.vectorized()
                        pred, prob, uncertainty= model.predict(skl_normalized)
                        #value = [pred, unc]
                        results[fps-1][gesture-1].append([pred,uncertainty])
                        if pred > 0:
                            unc += uncertainty
                            total+=1
                        else:
                            unc_spt += uncertainty
                            total_spt +=1

                        # time.sleep(0.2)
                        break
                    # print(pred, prob, uncertainty)
        #results[fps-1]=[unc/total, unc_spt/total_spt ]
        #print("mean uncertainty = ", unc/total, "acc:", 100*(correct/total))
    pickle.dump(results, open("results_complete.pkl","wb"))
    #plt.plot(results)
    #plt.legend(["Gesture","No Gesture"])
    #plt.show()

                
    

    