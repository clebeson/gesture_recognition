from collections import deque
import numpy as np
import cv2
import csv
import sys
import os
import argparse
import math
import pickle
import random
import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision import  models
import torch.optim as optim
import matplotlib.pyplot as plt
import logging
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable
import bayesian_layers as bl
import numpy as np 
from torch.optim import Adam
from timeit import default_timer as timer
import iteractive_starRGB_hand_dataset  as starRGB_dataset
import iteractive_starRGB_hand_model  as starRGB_model

import utils
from hyperopt import hp
torch.manual_seed(30)
np.random.seed(30)
random.seed(30) 



class RealTimeModel(nn.Module):
    def __init__(self,  rec_model, spt_model,window_size = 1):
        super(RealTimeModel, self).__init__()
        #type = bl.ModelType.VAR_DROP_B_ADAP
        self.classifier = rec_model
        self.classifier.eval()
        self.classifier.set_dropout(0.3)
        self.spotting = spt_model
        self.spotting.set_dropout(0.3)
        self.spotting.eval()
        self.window_size = window_size
        self.window = []
        self.movement = False

    def start_hidden(self):
        self.classifier.start_hidden()
        self.spotting.start_hidden()
        self.movement = False
        self.window = []


    def predict_spotting(self, mov, hands, mc_simulation = None):
        with torch.no_grad():
                out = self.spotting(mov,hands,mc_simulation)
                out = out.cpu()
                probs = F.log_softmax(out,1).exp().detach().numpy()
                mean = probs.mean(0)
                pred = np.argmax(mean)  
        return pred,probs


    def predict(self, mov, hands, mc_simulation = None):
        spp_pred, spp_prob, = self.predict_spotting(mov,hands,mc_simulation=mc_simulation)
        if len(self.window) == self.window_size: del self.window[0]
        self.window.append(spp_pred)
        w = sum(self.window)/self.window_size


        if w == 1.0 and not self.movement: 
            self.movement = True
        elif w == 0.0 and self.movement: 
            self.movement = False
        
        # print(w, spp_pred, self.movement)
        
        if not self.movement:
            self.classifier.start_hidden()
            return 0, spp_prob
        
        with torch.no_grad():
                out = self.classifier(mov,hands, mc_simulation)
                out = out.cpu()
                probs = F.log_softmax(out,1).exp().detach().numpy()
                mean = probs.mean(0)
                pred = np.argmax(mean) + 1    
        return pred, probs

    def _calc_uncertainty(self,probs):
        if len(probs.shape) > 2:
            mean = probs.mean(1)
            h = -(mean*np.log(mean)).sum(1) #entropy
        else: 
            mean = probs.mean(0)
            h = -(mean*np.log(mean)).sum(0) #entropy
            # s = probs.std(0).sum()
        return h

def do_rec_spt(path_spt, path_rec):
    start = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_rec,params_spt = starRGB_model.IterStarRGBHandModel.load_model(path_rec,"MC")
    model_spt,params_spt = starRGB_model.IterStarRGBHandModel.load_model(path_spt,"MC")
    model_rec.set_dropout(0.1)
    model_spt.set_dropout(0.1)
    rt_model = RealTimeModel(model_rec, model_spt)
    rt_model = rt_model.to(device)
    test_data = starRGB_dataset.IterStarRGBHandDataset(dataset="test",alpha = params_spt["alpha"], window = params_spt["window"], 
                                                    max_size = None, spotting = True,
                                                    transform=starRGB_dataset.DataTrasformation(output_size=(110,120), data_aug = False))
    results  = []
    corrects = 0
    total = 0
    samples = []
    rt_model.eval()
    for data in test_data:
        rt_model.start_hidden()
        mov,label, hands = data["images"].to(device).unsqueeze(0), data["label"].numpy().reshape(-1), data["hands"].to(device).unsqueeze(0)
        predictions = {"pred":[],"label":[],"probs":[],"file": data["file"], "weights_clf":[], "weights_spt":[]}
        samples.append(mov.size(1))
        # print(mov.shape, label.shape,hands.shape)
        for idx in range(mov.size(1)):
            m,l, h  = mov[:,idx],label[idx], hands[:,idx]
            m, h= m.unsqueeze(1), h.unsqueeze(1)
            # pred, probs = rt_model.predict_spotting(m, h, mc_simulation = 20)
            pred, probs = rt_model.predict(m,h, mc_simulation = 20)
            wc, ws = rt_model.classifier.soft.weights, rt_model.spotting.soft.weights
            predictions["pred"].append(pred)
            predictions["label"].append(l)
            predictions["probs"].append(probs)
            predictions["weights_clf"].append(None if wc is None else wc[0])
            predictions["weights_spt"].append(None if ws is None else ws[0])
            corrects += pred==l
            total += 1
        print(pred)
        print(l)
        results.append(predictions)
        acc = corrects/total
        print(acc)
        # break

    end = time.time()
    

    with open('results_anticipation_RTM_{}.pkl'.format( int(acc*10000)), 'wb') as file:
            pickle.dump(results, file, protocol=pickle.HIGHEST_PROTOCOL)
    utils.log("Anticipation RT model acc = {:.2f}".format(acc*100))
    samples = np.array(samples)
    utils.log("Elapsed = {}  mean = {}  std = {} mean time = {}".format(end-start, samples.mean(), samples.std(), samples.sum()/(end-start)))

if __name__ == "__main__":
    import time
    do_rec_spt("dynamic_star_rgb_hand_SPT_LSTM_9354.pth", "dynamic_star_rgb_hand_MCLSTM_9745.pth")
   