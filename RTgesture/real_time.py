from collections import deque
import numpy as np
import cv2
import csv
import sys
import time
import os
import argparse
import math
import glob
import pickle
import random
import torch
# from parallel import DataParallelModel, DataParallelCriterion
from torch.nn.parallel import DataParallel
import torch.nn as nn
from torch.nn import functional as F
from itertools import product
import torch.optim as optim
import matplotlib.pyplot as plt
import skimage as sk
import utils
import logging
import threading
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable
import bayesian_layers as bl
import random 
from skimage import io, transform
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from skeleton import *
from hand_data_transformation import HandTrasformation
from hand_skl_dataset import HandSKLDataset

torch.manual_seed(30)
np.random.seed(30)
random.seed(30) 




class SoftAttention(nn.Module):
    def __init__(self,  input_size = 512, hidden_size = 128, type = "det" ):
        super(SoftAttention, self).__init__()
        if type == "mc": self.type =  bl.ModelType.MC_DROP
        elif type == "vd": self.type = bl.ModelType.VAR_DROP_B_ADAP
        elif type == "bbb": self.type = bl.ModelType.BBB
        else: self.type = bl.ModelType.DET
        args = {
             "mu":0,
             "logstd1":1.0,
             "logstd2":1.0,
             "pi":0.5,
             "type":self.type,
             "dropout":0
             }
        self.softatt = nn.Sequential(bl.Linear(input_size, hidden_size, **args),
                                     nn.ReLU(),
                                     bl.Linear(hidden_size, 1, **args),
                                    )
        self.weights = None
        
    def forward(self,inputs):
        weights = []
        for x in inputs: weights.append(self.softatt(x))
        weights, inputs = torch.cat(weights,1), torch.cat([x_i.unsqueeze(1) for x_i in inputs],1)
        weights = F.log_softmax(weights,1).exp().unsqueeze(2)
        soft = torch.sum(weights*inputs,1)
        self.weights = weights.squeeze(2).data.cpu().numpy()
        return soft


class SqueezeExtractor(nn.Module):
    def __init__(self):
        super(SqueezeExtractor, self).__init__()
        
    def forward(self,x):
        return x.squeeze(2).squeeze(2)

class RTGR(nn.Module):
    
    def __init__(self, input_size, output_size, hidden_dim,n_layers, mode = "mc", dropout = 0.5, logstd1 = -1, logstd2 = -2, pi = 0.5 ):
        super(RTGR, self).__init__()
        #type = bl.ModelType.VAR_DROP_B_ADAP
        if mode == "mc": type =  bl.ModelType.MC_DROP
        elif mode == "vd": type = bl.ModelType.VAR_DROP_B_ADAP
        elif mode == "bbb": type = bl.ModelType.BBB
        else: type = bl.ModelType.DET
        
        linear_args =  {
            "mu":0,
             "logstd1":logstd1,
             "logstd2":logstd2,
             "pi":pi,
             "type":bl.ModelType.DET,
             "dropout":dropout
             }
        rnn_args =  {
            "mu":0,
             "logstd1":logstd1,
             "logstd2":logstd2,
             "pi":pi,
              "type":type,
             "dropout":dropout
             }
        last_linear_args = {
            "mu":0,
             "logstd1":logstd1,
             "logstd2":logstd2,
             "pi":pi,
             "type":type,
             "dropout":0
             }

        #Embedding
        self.skl_input_size = input_size
        self.skl_emb_size = 256

        self.hand_input_size = 512
        self.hand_emb_size = 256

        bfc1 = bl.Linear(self.skl_input_size, self.skl_emb_size, **linear_args)
        bfc2 = bl.Linear(self.skl_emb_size, self.skl_emb_size, **linear_args)

        bfc3 = bl.Linear(self.hand_input_size, self.hand_emb_size, **linear_args)
        # self.bfc4 = bl.Linear(512, self.hand_emb_size, **linear_args)
        


        self.hand_embedding = nn.Sequential(bfc3,nn.ReLU())
        self.skl_embedding = nn.Sequential(bfc1,nn.ReLU(), bfc2,nn.ReLU())
        self.soft_att = SoftAttention(input_size = self.hand_input_size)
        self.soft_att_info = SoftAttention(input_size = self.hand_emb_size)
        self.soft_att_spt = SoftAttention(input_size = self.hand_emb_size)


        model = models.resnet34(pretrained=True)
        features= list(model.children())
        
        
        self.hand_features = nn.Sequential(*features[:-1], SqueezeExtractor())


        weights = [1.0, 1.0]
        class_weights = torch.FloatTensor(weights).cuda()
        self.loss_fn = nn.NLLLoss(weight = class_weights)
        self.output_size = output_size
        self.hidden_dim = hidden_dim 
        self.n_layers = n_layers 
        self.rnn = bl.LSTM( input_size = self.skl_emb_size  ,\
                                    hidden_size = self.hidden_dim, num_layers = self.n_layers, \
                                    batch_first=True,**rnn_args)              
        self.bfc_output = bl.Linear(self.hidden_dim, 15, **last_linear_args)


        self.rnn_spt = bl.LSTM( input_size = self.skl_emb_size  ,\
                                    hidden_size = self.hidden_dim, num_layers = self.n_layers, \
                                    batch_first=True,**rnn_args)              
        self.bfc_output_spt = bl.Linear(self.hidden_dim, 2, **last_linear_args)
        
        self.dropout = dropout
        self._baysian_layers = []
        self.hidden = None
        self.hidden_spt = None
        self.moviment = False


    def get_spt_parameters(self):
        classifier = list(self.rnn_spt.parameters())
        classifier += self.bfc_output_spt.parameters()
        classifier += self.soft_att_spt.parameters()
        classifier += self.hand_embedding.parameters()
        classifier += self.skl_embedding.parameters()
        classifier += self.soft_att.parameters()
        features = list(self.hand_features.parameters())
        return features, classifier

    def freeze_classifier(self, freeze = True):
        ft,clf = self.get_parameters()
        for p in ft: p.requires_grad = not freeze
        for p in clf: p.requires_grad = not freeze

    def freeze_spotting(self, freeze = True):
        spt= self.get_spt_parameters()
        for p in spt: p.requires_grad = not freeze


    def get_parameters(self):
        classifier = list(self.rnn.parameters())
        classifier += self.bfc_output.parameters()
        classifier += self.hand_embedding.parameters()
        classifier += self.skl_embedding.parameters()
        classifier += self.soft_att.parameters()
        classifier += self.soft_att_info.parameters()
        features = list(self.hand_features.parameters())
        # features += self.conv_hand.parameters()
        return features, classifier

    def sampling(self,sampling,store):
        for layer in self._baysian_layers:
            layer.sampling = sampling
            layer.store_samples = store

    def forward(self, hand, skl, mc_samples = None):
        if mc_samples:
            hand = hand.repeat((mc_samples,1, 1, 1, 1,1))
            skl = skl.repeat((mc_samples,1,1))

        batch_size, seq_size,N,C,H,W = hand.size()
        hand = hand.reshape(-1,N,C,H,W)
        hand_l = self.hand_features(hand[:,0])
        hand_r = self.hand_features(hand[:,1])
        skl= skl.reshape( batch_size*seq_size,-1)
        skl_emb = self.skl_embedding(skl)
        hand = self.soft_att([hand_l, hand_r])
        hand_emb = self.hand_embedding(hand)
        x = self.soft_att_spt([hand_emb, skl_emb])
        x = x.view(batch_size, seq_size,-1)
        out, hidden = self.rnn_spt(x, self.hidden_spt)
        self.hidden_spt = ([h.data for h in hidden])
        out = self.bfc_output_spt(out) 
        out = out.contiguous().view(-1, out.size(-1))
        prob = F.log_softmax(out,1).exp()
        spt_pred = prob.mean(0).argmax()

        if spt_pred == 1:
            if not self.moviment:
                self.moviment = True
                self.hidden = None
            x = self.soft_att_info([hand_emb, skl_emb])
            x = x.view(batch_size, seq_size,-1)
            out, hidden = self.rnn(x, self.hidden)
            self.hidden = ([h.data for h in hidden])
            out = self.bfc_output(out) 
            out = out.contiguous().view(-1, out.size(-1))
            prob = F.log_softmax(out,1).exp()
        else:
            self.moviment = False
        return prob


    def set_dropout(self, value, training=True):
        for layer in self.get_baysian_layers(): 
            layer.dropout = value


    def get_baysian_layers(self):
        if not self._baysian_layers:
            self._baysian_layers = [module for module in self.modules() if ((isinstance(module,bl.BaseLayer) or isinstance(module,bl.BaseRNNLayer))  and module._type !=  bl.ModelType.DET) ]
        return self._baysian_layers

    def reset_hidden(self):
        self.hidden = None

    def sharpening_posterior(self, x, hidden, outputs, target):
        # We compute the cost -  L = -log p(y|x,w) 
        NLL = self.get_nll(outputs, target)
        
        # The gradients of nll with respect to lstm Mu and Sigma
        gradients = torch.autograd.grad(outputs=NLL, inputs=self.lstm.weights, grad_outputs=torch.ones(NLL.size()).to(x.device), create_graph=True, retain_graph=True, only_inputs=True)
        
        # Then we do the forward pass again with sharpening:
        output, hidden = self.lstm(x, hidden, gradients)
        return output, hidden

    def parallelize(self):
        model = DataParallel(self)
        # self.loss_fn = DataParallelCriterion(self.loss_fn)
        return model

    def get_nll(self, output, targets):
        """
        return:
            NLL: Negative Loglikelihood Loss
        """
        return self.loss_fn(output, targets)


    def get_loss(self, output, targets, batch):
        """
        return:
            NLL: NLL is averaged over seq_len and batch_size
            KL: KL is the original scale KL
        """
        # NLL (classification loss)
        NLL = self.get_nll(output, targets)
        

        # KL divergence between posterior and variational prior distribution
        KL =  Variable(torch.zeros(1)).to(targets.device) 
        
        for layer in self.get_baysian_layers(): 
            KL += layer.get_kl()
        
        return NLL, KL/batch  #output.size(32)


                
def predict(modelo, data, device, results):
    mc_samples = 20 #amount of samples for Monte Carlo Estimation
    with torch.no_grad():
        hand_seq, skl_seq, video_label = data["images"], data["skeletons"], data["labels"]
        # label = video_label.repeat(mc_samples,1).reshape(-1).long()
        for i in range(hand_seq.shape[1]):
            hand = hand_seq[:,i].unsqueeze(1)
            skl = skl_seq[:,i].unsqueeze(1)
            skl = skl.to(device)
            hand = hand.to(device)
            probs = model(hand,skl,mc_samples).cpu().numpy()
            pred = np.argmax(probs.mean(0))
            results["pred"].append(pred)
            results["probs"].append(probs)
            results["label"].append(video_label[0,i].numpy())

            
if __name__ == "__main__":
    results = {"pred":[], "label":[],  "probs":[]}
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    dict_save = torch.load("saved_models/model_gesture_spt_8797.00.pth")
    hidden_dim = dict_save["hidden_dim"]
    n_layers = dict_save["num_layers"]
    model = RTGR( 21, 15, hidden_dim,n_layers, mode = "mc", dropout = 0.1)
    model = model.to(device)
    model.load_state_dict(dict_save["model"],strict= True)
    model.eval()
    data_test = HandSKLDataset(type = "test", max_seq=  100, \
                        transform=HandTrasformation(output_size = (40,40), data_aug=False), spotting=2)
    test_loader = DataLoader(data_test, \
            batch_size=1, \
            pin_memory = True,\
            shuffle=False, \
            num_workers=5,\
            drop_last = False)

    for samples in test_loader:
        predict(model,samples,device,results)


    results["pred"] = np.array(results["pred"])
    results["label"]= np.array(results["label"])
    # results["probs"]= np.array(results["probs"])



    acc = sum(results["pred"] == results["label"])/len(results["pred"])
    print(acc)
    pickle.dump(results, open( "predictions/prediction_{}_{:.2f}.pkl".format("gesture_real_timeC",int(acc*10000)), "wb" ), protocol=2)
    
            


