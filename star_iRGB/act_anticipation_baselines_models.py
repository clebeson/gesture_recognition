from collections import deque
import numpy as np
import cv2
import csv
import sys
import os
import argparse
import math
import glob
import pickle
import random
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, models, datasets
import torch.optim as optim
import matplotlib.pyplot as plt
import logging
import threading
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable
import bayesian_layers as bl
from hmmlearn import hmm
import numpy as np 
from scipy.stats import multivariate_normal
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC, NuSVC
import hyperopt
import requests
from torch.optim import Adam
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval
import math
import sys
import collections
import itertools
import matplotlib.pyplot as plt
from scipy.stats import mode
from scipy.spatial.distance import squareform
import warnings
from sklearn.exceptions import ConvergenceWarning
import random
from bottleneck_dataset import *
from dynamic_star_dataset import *
from timeit import default_timer as timer
warnings.filterwarnings(action='ignore', category=ConvergenceWarning)
torch.manual_seed(30)
np.random.seed(30)
random.seed(30) 

class DStarRGBSpottingModel(nn.Module):
    def __init__(self,  output_size, hidden_dim,n_layers, mode = "mc", dropout = 0.5, logstd1 = -1, logstd2 = -2, pi = 0.5 ):
        super(DStarRGBSpottingModel, self).__init__()
        #type = bl.ModelType.VAR_DROP_B_ADAP
        if mode == "mc": self.type =  bl.ModelType.MC_DROP
        elif mode == "vd": self.type = bl.ModelType.VAR_DROP_B_ADAP
        elif mode == "bbb": self.type = bl.ModelType.BBB
        else: self.type = bl.ModelType.DET
        
        linear_args =  {
             "mu":0,
             "logstd1":logstd1,
             "logstd2":logstd2,
             "pi":pi,
             "type":self.type,
             "dropout":dropout
             }
        rnn_args =  {
            "mu":0,
             "logstd1":logstd1,
             "logstd2":logstd2,
             "pi":pi,
              "type":self.type,
             "dropout":dropout
             }
        last_linear_args = {
            "mu":0,
             "logstd1":logstd1,
             "logstd2":logstd2,
             "pi":pi,
             "type":self.type,
             "dropout":0
             }

       
        self.conv1d =nn.Conv1d(1, 1, 3, stride=2)
        self.embeding_size = 1023
        weights = [1.0, 1.74]

        class_weights = torch.FloatTensor(weights).cuda()
        self.loss_fn = nn.NLLLoss(weight = class_weights)  if output_size == 2 else nn.NLLLoss()
        self.output_size = output_size #12
        self.hidden_dim = hidden_dim #256
        self.n_layers = n_layers #2
        self.sharpen = False
        self.lstm = bl.LSTM( input_size = self.embeding_size,\
                                    hidden_size = self.hidden_dim, num_layers = self.n_layers, \
                                    batch_first=True,**rnn_args) 
                                    
        
        self.dropout = dropout
        self.fc = bl.Linear(self.hidden_dim, self.output_size, **last_linear_args)
        self.dp_training = True
        self._baysian_layers = []


    def get_parameters(self):
        classifier = list(self.lstm.parameters())
        classifier += self.conv1d.parameters()
        classifier += self.fc.parameters()
        return classifier
        

    def sampling(self,sampling,store):
        for layer in self._baysian_layers:
            layer.sampling = sampling
            layer.store_samples = store

    def forward(self, x, hidden):
        size = x.size()
        x = x.contiguous().view(-1,1,size[-1])
        x = self.conv1d(x)
        x = x.contiguous().view(size[0],size[1],-1)
        out, hidden_out = self.lstm(x, hidden)
        out = self.fc(out) 
        out = out.contiguous().view(-1, out.size(-1))
        return out, hidden_out

    def set_dropout(self, value, training=True):
        self.dropout = value
        for layer in self.get_baysian_layers(): layer.dropout = value


    def sharpening_posterior(self, x, hidden, outputs, target):
        # We compute the cost -  L = -log p(y|x,w) 
        NLL = self.get_nll(outputs, target)
        
        # The gradients of nll with respect to lstm Mu and Sigma
        gradients = torch.autograd.grad(outputs=NLL, inputs=self.lstm.weights, grad_outputs=torch.ones(NLL.size()).to(x.device), create_graph=True, retain_graph=True, only_inputs=True)
        
        # Then we do the forward pass again with sharpening:
        output, hidden = self.lstm(x, hidden, gradients)
        return output, hidden


    def get_nll(self, output, targets):
        """
        return:
            NLL: Negative Loglikelihood Loss
        """
        return self.loss_fn(F.log_softmax(output,1), targets)


    def get_baysian_layers(self):
        if not self._baysian_layers:
            self._baysian_layers = [module for module in self.modules() if isinstance(module,bl.BaseLayer)]
        return self._baysian_layers


    def get_loss(self, output, targets):
        """
        return:
            NLL: NLL is averaged over seq_len and batch_size
            KL: KL is the original scale KL
        """
        NLL = self.get_nll(output, targets)
        return NLL, 0

class SoftAttention(nn.Module):
    def __init__(self,  input_size = 512, hidden_size = 128, type = "DET" ):
        super(SoftAttention, self).__init__()
        if mode == "mc": self.type =  bl.ModelType.MC_DROP
        elif mode == "vd": self.type = bl.ModelType.VAR_DROP_B_ADAP
        elif mode == "bbb": self.type = bl.ModelType.BBB
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
        
    def forward(self,x1,x2):
        w1 = self.softatt(x1)
        w2 = self.softatt(x2)
        w = torch.cat([w1,w2],1)
        w = F.log_softmax(w,1).exp()
        soft = w[:,0]*x1.squeeze(1) + w[:,1]*x2.squeeze(1)
        return soft

class SqueezeExtractor(nn.Module):
    def __init__(self):
        super(SqueezeExtractor, self).__init__()
        
    def forward(self,x):
        return x.squeeze().unsqueeze(1)

class DStarRGBHandModel(nn.Module):
    def __init__(self,  output_size, hidden_dim,n_layers, mode = "mc", dropout = 0.5, logstd1 = -1, logstd2 = -2, pi = 0.5 ):
        super(DStarRGBHandModel, self).__init__()
        #type = bl.ModelType.VAR_DROP_B_ADAP
        if mode == "mc": self.type =  bl.ModelType.MC_DROP
        elif mode == "vd": self.type = bl.ModelType.VAR_DROP_B_ADAP
        elif mode == "bbb": self.type = bl.ModelType.BBB
        else: self.type = bl.ModelType.DET
        
        linear_args =  {
            "mu":0,
             "logstd1":logstd1,
             "logstd2":logstd2,
             "pi":pi,
             "type":self.type,
             "dropout":dropout
             }
        rnn_args =  {
            "mu":0,
             "logstd1":logstd1,
             "logstd2":logstd2,
             "pi":pi,
              "type":self.type,
             "dropout":dropout
             }
        last_linear_args = {
            "mu":0,
             "logstd1":logstd1,
             "logstd2":logstd2,
             "pi":pi,
             "type":self.type,
             "dropout":0
             }

        resnet_mov = models.resnet50(pretrained=True)
        self.mov= nn.Sequential(*(list(resnet_mov.children())[:-1]),SqueezeExtractor(), nn.Conv1d(1, 1, 3, stride=2),nn.ReLU())
        resnet_hand = models.resnet50(pretrained=True)
        self.hand= nn.Sequential(*(list(resnet_hand.children())[:-1]), SqueezeExtractor(), nn.Conv1d(1, 1, 3, stride=2),nn.ReLU())
        #self.norm = nn.InstanceNorm1d(input_size)
        #Embedding
        self.embeding_size = 1023
        self.soft = SoftAttention(1023,128)

        # nn.MaxPool1d(2, stride=2)
        weights = [1./1.5, 1.0]

        class_weights = torch.FloatTensor(weights).cuda()
        self.loss_fn = nn.NLLLoss(weight = class_weights)  if output_size == 2 else nn.NLLLoss()
        self.output_size = output_size #12
        self.hidden_dim = hidden_dim #256
        self.n_layers = n_layers #2
        self.sharpen = False
        self.lstm = bl.LSTM( input_size = self.embeding_size,\
                                    hidden_size = self.hidden_dim, num_layers = self.n_layers, \
                                    batch_first=True,**rnn_args) 
                                    
        # self.noise = GaussianNoise(0.01)


        # dropout layer
        # self.norm1 = nn.BatchNorm1d(self.input_size)
        # self.norm2 = nn.BatchNorm1d(self.embeding_size)
        self.dropout = dropout
        # self.fc_combine = bl.Linear(self.input_size, self.input_size, **linear_args)
        # self.combine = nn.Sequential(self.fc_combine, nn.ReLU())
        
        # linear and sigmoid layers
        self.fc = bl.Linear(self.hidden_dim, self.output_size, **last_linear_args)
        self.dp_training = True
        self._baysian_layers = []
        self.hidden = None


    def get_parameters(self):
        extractor = list(self.mov.parameters())
        extractor += self.hand.parameters()
        classifier = list(self.soft.parameters())
        classifier += self.lstm.parameters()
        classifier += self.fc.parameters()
        return extractor, classifier
        
    def start_hidden(self):
        self.hidden = None

    def sampling(self,sampling,store):
        for layer in self._baysian_layers:
            layer.sampling = sampling
            layer.store_samples = store

    def forward(self, m, h):
        size_m, size_h = m.size(), h.size()

        m = m.contiguous().view(-1,size_m[-3],size_m[-2],size_m[-1])
        h = h.contiguous().view(-1,size_h[-3],size_h[-2],size_h[-1])

        m = self.mov(m)
        h = self.hand(h)
        x = self.soft(m,h)
        
        x = x.view(size_m[0],size_m[1],-1)
        
        
        out, hidden_out = self.lstm(x, self.hidden)
        self.hidden = ([h.data for h in hidden_out])

        out = self.fc(out) 
        out = out.contiguous().view(-1, out.size(-1))
        return out

    def set_dropout(self, value, training=True):
        self.dropout = value
        for layer in self.get_baysian_layers(): layer.dropout = value


    def sharpening_posterior(self, x, hidden, outputs, target):
        # We compute the cost -  L = -log p(y|x,w) 
        NLL = self.get_nll(outputs, target)
        
        # The gradients of nll with respect to lstm Mu and Sigma
        gradients = torch.autograd.grad(outputs=NLL, inputs=self.lstm.weights, grad_outputs=torch.ones(NLL.size()).to(x.device), create_graph=True, retain_graph=True, only_inputs=True)
        
        # Then we do the forward pass again with sharpening:
        output, hidden = self.lstm(x, hidden, gradients)
        return output, hidden


    def get_nll(self, output, targets):
        """
        return:
            NLL: Negative Loglikelihood Loss
        """
        return self.loss_fn(F.log_softmax(output,1), targets)


    def get_baysian_layers(self):
        if not self._baysian_layers:
            self._baysian_layers = [module for module in self.modules() if isinstance(module,bl.BaseLayer)]
        return self._baysian_layers


    def get_loss(self, output, targets):
        """
        return:
            NLL: NLL is averaged over seq_len and batch_size
            KL: KL is the original scale KL
        """
        # NLL (classification loss)
        NLL = self.get_nll(output, targets)
        return NLL, None

        # # KL divergence between posterior and variational prior distribution
        # KL =  Variable(torch.zeros(1)).to(output.device) 
        
        # for layer in self.get_baysian_layers(): 
        #     KL += layer.get_kl()
        
        # return NLL, KL/batch  #output.size(32)
        

class DStarRGBHandSpottingModel(nn.Module):
    def __init__(self,  output_size, hidden_dim,n_layers, mode = "mc", dropout = 0.5, logstd1 = -1, logstd2 = -2, pi = 0.5 ):
        super(DStarRGBHandSpottingModel, self).__init__()
        #type = bl.ModelType.VAR_DROP_B_ADAP
        if mode == "mc": self.type =  bl.ModelType.MC_DROP
        elif mode == "vd": self.type = bl.ModelType.VAR_DROP_B_ADAP
        elif mode == "bbb": self.type = bl.ModelType.BBB
        else: self.type = bl.ModelType.DET
        
        linear_args =  {
            "mu":0,
             "logstd1":logstd1,
             "logstd2":logstd2,
             "pi":pi,
             "type":self.type,
             "dropout":dropout
             }
        rnn_args =  {
            "mu":0,
             "logstd1":logstd1,
             "logstd2":logstd2,
             "pi":pi,
              "type":self.type,
             "dropout":dropout
             }
        last_linear_args = {
            "mu":0,
             "logstd1":logstd1,
             "logstd2":logstd2,
             "pi":pi,
             "type":self.type,
             "dropout":0
             }

        resnet_mov = models.resnet50(pretrained=True)
        self.mov= nn.Sequential(*(list(resnet_mov.children())[:-1]),SqueezeExtractor(), nn.Conv1d(1, 1, 3, stride=2),nn.ReLU())
        resnet_hand = models.resnet50(pretrained=True)
        self.hand= nn.Sequential(*(list(resnet_hand.children())[:-1]), SqueezeExtractor(), nn.Conv1d(1, 1, 3, stride=2),nn.ReLU())
        #self.norm = nn.InstanceNorm1d(input_size)
        #Embedding
        self.embeding_size = 1023
        self.soft = SoftAttention(1023,128)

        # nn.MaxPool1d(2, stride=2)
        weights = [1./1.5, 1.0]

        class_weights = torch.FloatTensor(weights).cuda()
        self.loss_fn = nn.NLLLoss(weight = class_weights)  if output_size == 2 else nn.NLLLoss()
        self.output_size = output_size #12
        self.hidden_dim = hidden_dim #256
        self.n_layers = n_layers #2
        self.sharpen = False
        self.lstm_spt = bl.LSTM( input_size = self.embeding_size,\
                                    hidden_size = self.hidden_dim, num_layers = self.n_layers, \
                                    batch_first=True,**rnn_args) 
                                    
        # self.noise = GaussianNoise(0.01)


        # dropout layer
        # self.norm1 = nn.BatchNorm1d(self.input_size)
        # self.norm2 = nn.BatchNorm1d(self.embeding_size)
        self.dropout = dropout
        # self.fc_combine = bl.Linear(self.input_size, self.input_size, **linear_args)
        # self.combine = nn.Sequential(self.fc_combine, nn.ReLU())
        
        # linear and sigmoid layers
        self.fc_spt = bl.Linear(self.hidden_dim, self.output_size, **last_linear_args)
        self.dp_training = True
        self._baysian_layers = []
        self.hidden = None


    def get_parameters(self):
        extractor = list(self.mov.parameters())
        extractor += self.hand.parameters()
        classifier = list(self.soft.parameters())
        classifier += self.lstm_spt.parameters()
        classifier += self.fc_spt.parameters()
        return extractor, classifier
        
    def start_hidden(self):
        self.hidden = None

    def sampling(self,sampling,store):   
        for layer in self._baysian_layers:
            layer.sampling = sampling
            layer.store_samples = store

    def forward(self, m, h):
        size_m, size_h = m.size(), h.size()

        m = m.contiguous().view(-1,size_m[-3],size_m[-2],size_m[-1])
        h = h.contiguous().view(-1,size_h[-3],size_h[-2],size_h[-1])

        m = self.mov(m)
        h = self.hand(h)
        x = self.soft(m,h)
        
        x = x.view(size_m[0],size_m[1],-1)
        
        
        out, hidden_out = self.lstm_spt(x, self.hidden)
        self.hidden = ([h.data for h in hidden_out])

        out = self.fc_spt(out) 
        out = out.contiguous().view(-1, out.size(-1))
        return out

    def set_dropout(self, value, training=True):
        self.dropout = value
        for layer in self.get_baysian_layers(): layer.dropout = value


    def sharpening_posterior(self, x, hidden, outputs, target):
        # We compute the cost -  L = -log p(y|x,w) 
        NLL = self.get_nll(outputs, target)
        
        # The gradients of nll with respect to lstm Mu and Sigma
        gradients = torch.autograd.grad(outputs=NLL, inputs=self.lstm.weights, grad_outputs=torch.ones(NLL.size()).to(x.device), create_graph=True, retain_graph=True, only_inputs=True)
        
        # Then we do the forward pass again with sharpening:
        output, hidden = self.lstm(x, hidden, gradients)
        return output, hidden


    def get_nll(self, output, targets):
        """
        return:
            NLL: Negative Loglikelihood Loss
        """
        return self.loss_fn(F.log_softmax(output,1), targets)


    def get_baysian_layers(self):
        if not self._baysian_layers:
            self._baysian_layers = [module for module in self.modules() if isinstance(module,bl.BaseLayer)]
        return self._baysian_layers


    def get_loss(self, output, targets):
        """
        return:
            NLL: NLL is averaged over seq_len and batch_size
            KL: KL is the original scale KL
        """
        # NLL (classification loss)
        NLL = self.get_nll(output, targets)
        return NLL, None

        # # KL divergence between posterior and variational prior distribution
        # KL =  Variable(torch.zeros(1)).to(output.device) 
        
        # for layer in self.get_baysian_layers(): 
        #     KL += layer.get_kl()
        
        # return NLL, KL/batch  #output.size(32)





class DStarRGBModel(nn.Module):
    def __init__(self,  output_size, hidden_dim,n_layers, mode = "mc", dropout = 0.5, logstd1 = -1, logstd2 = -2, pi = 0.5 ):
        super(DStarRGBModel, self).__init__()
        #type = bl.ModelType.VAR_DROP_B_ADAP
        if mode == "mc": self.type =  bl.ModelType.MC_DROP
        elif mode == "vd": self.type = bl.ModelType.VAR_DROP_B_ADAP
        elif mode == "bbb": self.type = bl.ModelType.BBB
        else: self.type = bl.ModelType.DET
        
        linear_args =  {
            "mu":0,
             "logstd1":logstd1,
             "logstd2":logstd2,
             "pi":pi,
             "type":self.type,
             "dropout":dropout
             }
        rnn_args =  {
            "mu":0,
             "logstd1":logstd1,
             "logstd2":logstd2,
             "pi":pi,
              "type":self.type,
             "dropout":dropout
             }
        last_linear_args = {
            "mu":0,
             "logstd1":logstd1,
             "logstd2":logstd2,
             "pi":pi,
             "type":self.type,
             "dropout":0
             }

        resnet = models.resnet50(pretrained=True)
        self.resnet= nn.Sequential(*(list(resnet.children())[:-1]))
        #self.norm = nn.InstanceNorm1d(input_size)
        #Embedding
        self.conv1d =nn.Conv1d(1, 1, 3, stride=2)
        # nn.MaxPool1d(2, stride=2)
        self.embeding_size = 1023
        weights = [1./1.5, 1.0]

        class_weights = torch.FloatTensor(weights).cuda()
        self.loss_fn = nn.NLLLoss(weight = class_weights)  if output_size == 2 else nn.NLLLoss()
        self.output_size = output_size #12
        self.hidden_dim = hidden_dim #256
        self.n_layers = n_layers #2
        self.sharpen = False
        self.lstm = bl.LSTM( input_size = self.embeding_size,\
                                    hidden_size = self.hidden_dim, num_layers = self.n_layers, \
                                    batch_first=True,**rnn_args) 
                                    
        # self.noise = GaussianNoise(0.01)


        # dropout layer
        # self.norm1 = nn.BatchNorm1d(self.input_size)
        # self.norm2 = nn.BatchNorm1d(self.embeding_size)
        self.dropout = dropout
        # self.fc_combine = bl.Linear(self.input_size, self.input_size, **linear_args)
        # self.combine = nn.Sequential(self.fc_combine, nn.ReLU())
        
        # linear and sigmoid layers
        self.fc = bl.Linear(self.hidden_dim, self.output_size, **last_linear_args)
        self.dp_training = True
        self._baysian_layers = []
        self.hidden = None

    def start_hidden(self):
        self.hidden = None

    def get_parameters(self):
        classifier = list(self.lstm.parameters())
        classifier += self.conv1d.parameters()
        classifier += self.fc.parameters()
        return self.resnet.parameters(), classifier
        

    def sampling(self,sampling,store):
        for layer in self._baysian_layers:
            layer.sampling = sampling
            layer.store_samples = store

    def forward(self, x):
        size = x.size()
        x = x.contiguous().view(-1,size[-3],size[-2],size[-1])
        x = self.resnet(x)
        x = x.squeeze().unsqueeze(1)
        x = self.conv1d(x).squeeze()
        x = x.view(size[0],size[1],-1)

        out, hidden_out = self.lstm(x, self.hidden)
        self.hidden = ([h.data for h in hidden_out])

        out = self.fc(out) 
        
        out = out.contiguous().view(-1, out.size(-1))
        return out

    def set_dropout(self, value, training=True):
        self.dropout = value
        for layer in self.get_baysian_layers(): layer.dropout = value


    def sharpening_posterior(self, x, hidden, outputs, target):
        # We compute the cost -  L = -log p(y|x,w) 
        NLL = self.get_nll(outputs, target)
        
        # The gradients of nll with respect to lstm Mu and Sigma
        gradients = torch.autograd.grad(outputs=NLL, inputs=self.lstm.weights, grad_outputs=torch.ones(NLL.size()).to(x.device), create_graph=True, retain_graph=True, only_inputs=True)
        
        # Then we do the forward pass again with sharpening:
        output, hidden = self.lstm(x, hidden, gradients)
        return output, hidden


    def get_nll(self, output, targets):
        """
        return:
            NLL: Negative Loglikelihood Loss
        """
        return self.loss_fn(F.log_softmax(output,1), targets)


    def get_baysian_layers(self):
        if not self._baysian_layers:
            self._baysian_layers = [module for module in self.modules() if isinstance(module,bl.BaseLayer)]
        return self._baysian_layers


    def get_loss(self, output, targets):
        """
        return:
            NLL: NLL is averaged over seq_len and batch_size
            KL: KL is the original scale KL
        """
        # NLL (classification loss)
        NLL = self.get_nll(output, targets)
        return NLL, None

        # # KL divergence between posterior and variational prior distribution
        # KL =  Variable(torch.zeros(1)).to(output.device) 
        
        # for layer in self.get_baysian_layers(): 
        #     KL += layer.get_kl()
        
        # return NLL, KL/batch  #output.size(32)
        

def create_datasets(num_workers = 2, batch_size = 32, max_size = 32, alpha = 0.7, window = 5, seq = 64):

    image_datasets = {
        "train":DStarRGBDataset(dataset="train", max_size = max_size, alpha = alpha, window = window, 
                            transform=DataTrasformation(output_size=(110,120), data_aug = True)),
        "test":DStarRGBDataset(dataset="test", max_size = max_size, alpha = alpha, window = window, 
                            transform=DataTrasformation(output_size=(110,120), data_aug = False)),
        "val":DStarRGBDataset(dataset="validation", max_size = max_size, alpha = alpha, window = window,
                            transform=DataTrasformation(output_size=(110,120), data_aug = False))
                }

    dataloaders = {
        "train":DataLoader(image_datasets["train"], 
                                            batch_size=batch_size, pin_memory=True, shuffle=True, 
                                            num_workers=num_workers, drop_last=True),

        # "test":image_datasets["test"],

        "test":DataLoader(image_datasets["test"], 
                                batch_size=30, pin_memory=True, shuffle=False, 
                                num_workers=num_workers, drop_last=False),

        "val":DataLoader(image_datasets["val"], 
                                batch_size=48, pin_memory=True, shuffle=False, 
                                  num_workers=num_workers, drop_last=True),
                 }
    return dataloaders                              
    

def create_spt_dataset(num_workers = 2, batch_size = 32, max_size = 32):

    dataloaders = {
                   "train":DataLoader(BottleneckDataset(dataset="train", sequence = max_size), 
                                    batch_size=batch_size, pin_memory=True, shuffle=True, 
                                    num_workers=num_workers, drop_last=False),
                    "test":BottleneckDataset(dataset="test", sequence = 50),
                    "val":BottleneckDataset(dataset="validation", sequence = 50)   
                   }
    return dataloaders

#Truncate long sequences and pad the small ones with relation to the parameter 'max'
def _padding(videos,max):
    sizes = []
    [sizes.append(len(v)) for v in videos]
    sizes = np.array(sizes)
    padded_data = np.ones((len(videos),max,len(videos[0][0][0])))*-1
    padded_labels = np.ones((len(videos),max))
    
    for i,(video,label) in enumerate(videos):
        padded_labels[i] = label
        if len(video) > max:
            video = video[:max]
        padded_data[i,:len(video)] = video
        #padded_data[i,-len(video):] = video[-1,1:]
        # print(padded_data[i].sum()- video[:,1:].sum())

    padded_data = padded_data.astype(float)
    padded_labels = padded_labels.astype(int)
    return padded_data, padded_labels

#Produce batchs for each step
def to_batch(videos, batch_size = 32, seq = 1, max = 100):
        indexes = list(range(len(videos)))
        random.shuffle(indexes)
        videos = [videos[i] for i in indexes]
        for b in range(len(videos)//batch_size):
            video_batch = [videos[i] for i in range(b*batch_size,(b+1)*batch_size)]
            
            padded_data, padded_labels = _padding(video_batch,max= max)
            size = padded_data.shape[1] // seq
            for s in range(size):
                label = padded_labels[:,s*seq:(s+1)*seq]
                data = padded_data[:,s*seq:(s+1)*seq]
                # data = data[:,:,[0, 1, 2,3,30,31,32,33,34,35,36,37, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,38,39,40,41]]
                yield data , label , True if s == (size-1) else False

def save_bottleneack(model, file):
    def crop_center( image, out):
        y,x = image.shape[1], image.shape[2]
        cropy, cropx = out
        startx = x//2-(cropx//2)
        starty = y//2-(cropy//2) 
        startx = 0 if startx + cropx >=x  else startx
        starty = 0 if starty + cropy >=y else starty
        return image[:,starty:starty+cropy,startx:startx+cropx,:]

    data = np.load(file)
    images, labels = data["images"], data["labels"]
    images = crop_center(images, (110,120))
    images = np.transpose(images, (0,3,1,2)).astype(np.float32)/255.0

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    results = []
    batch = 256
    steps = len(images)//batch

    with torch.no_grad():
        for step in range(steps):
            b = step*batch
            e = b + batch if step < steps-1 else len(images)
            batch_image = torch.from_numpy(images[b:e]).to(device)
            out = model(batch_image,None)
            results.append(out.squeeze().cpu().detach().numpy())
    data = np.concatenate(results,0)
    print(file,data.shape, labels.shape)
    np.savez(file.replace("numpy_files","bottleneck"),images = data,labels = labels)
           


def predict_DLSTM(model, test_data, params, results):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    running_corrects = 0
    total = 0
    with torch.no_grad():
        for data in test_data:
            # data,label, hands = data["images"].to(device), data["label"].numpy(), data["hands"].to(device)
            data, label = data["images"].to(device), data["label"]
            if len(data.shape) < 5: data = data.unsqueeze(0)
            # if len(hands.shape) < 5: hands = hands.unsqueeze(0)
            if type(model) == nn.DataParallel:
                model.module.start_hidden()
            else:
                model.start_hidden()
            out = model(data)
            out = out.cpu()
            probs = F.log_softmax(out,1).exp().detach().numpy()
            probs = probs.reshape((data.size(0),data.size(1),-1))[:,-1,:] #take the last prediction from each sequence
            # probs = probs.reshape((data.size(0),data.size(1),-1))#spotting
            label = label.reshape(-1).numpy()
            pred = np.argmax(probs, 1)
            running_corrects += np.sum(pred == label)
            results.append({"pred":pred, "label":label, "probs":probs})
            total+= len(pred)
            
    return running_corrects/total  

def predict_BBBLSTM(model, videos, params, results):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_class = params["num_classes"]
    model.to(device)
    model.train()
    # model.set_dropout(0.2)
    running_corrects = 0
    mc_samples = 20
    
    with torch.no_grad():
        for (video, label, interval) in videos:
            probs = np.zeros((mc_samples, len(video, num_class)))
            hiddens = [None for _ in range(mc_samples)]
            for i, data in enumerate(video):
                data = torch.from_numpy(data).float().unsqueeze(0)
                data = data.to(device)
                for mc in range(mc_samples):
                    out, hidden = model(data,hiddens[mc])
                    hiddens[mc] = [h.data for h in hidden]
                    # print(out.shape)
                    probs[mc,i] = F.log_softmax(out,1).exp().detach().numpy()
            pred = np.argmax(probs.mean(0), 1)[-1]
            running_corrects += np.sum(pred == label)
            results.append({"pred":pred, "label":label,  "probs":probs, "interval":interval})
    return running_corrects/len(videos)  


def predict_BLSTM(model, videos, params, results):
    device = torch.device('cuda:{}'.format(params["gpu"]) if torch.cuda.is_available() else 'cpu')
    num_class = params["num_classes"]
    model.to(device)
    model.train()
    model.set_dropout(0.1)
    running_corrects = 0
    mc_samples = 20
    with torch.no_grad():
        for (video, label, interval) in videos:
            data = np.repeat(np.expand_dims(video,0),mc_samples,0)
            data = torch.from_numpy(data).float()
            data = data.to(device)
            out, _ = model(data,None)
            out = out.view(mc_samples,len(video),-1).cpu()
            probs = F.log_softmax(out,1).exp().detach().numpy()
            pred = np.argmax(probs.mean(0), 1)[-1]
            running_corrects += np.sum(pred == label)
            results.append({"pred":pred, "label":label,  "probs":probs, "interval":interval})
    return running_corrects/len(videos)   

def train_LSTM( model,train_data, val_data, params):
    device = torch.device('cuda:{}'.format(params["gpu"]) if torch.cuda.is_available() else 'cpu')
    if model is None:
        model = DStarRGBHandModel( 
                    output_size =  params["num_classes"], 
                    hidden_dim =  params["hidden_dim"], 
                    n_layers =  params["n_layers"],  
                    mode =  "DET",
                    dropout = params["fc_drop"], 
                    )

    model = model.to(device)
    model = nn.DataParallel(model)
    model.train()
    # model.set_dropout(0.0)
    #Hyperparameters
    epochs = params["epoch"]
    batch = params["batch_size"]
    sequence = params["max_seq"][0] 
    max_sequence = params["max_seq"][1] 
    clip = params["clip"]
    lr=params["lr"]
    

    ft, clf = model.module.get_parameters()
    optimizers = [ optim.Adam(ft, lr=lr[0]), optim.Adam(clf, lr=lr[1])]
    schedulers = [StepLR(optimizers[0], step_size = 1, gamma = 0.99),
                  StepLR(optimizers[1], step_size = 1, gamma = 0.99) ]
    # clf = model.parameters()
    # optimizers = [optim.Adam(clf, lr=lr[1])]
    # schedulers = [StepLR(optimizers[0], step_size = 1, gamma = 0.99) ]

    acc_val = 0
    best_acc_val = 0
    accuracies = []
    B = len(train_data)//batch
    C = params["max_seq"][1] //sequence
    loss = 0
    for epoch in range(epochs):
        entered = False
        running_loss = 0
        running_kl = 0
        running_corrects = 0 
        total = 1e-18
        #importance of KL-divergence terms in relation to classification loss
        scale = 0.1
        model.module.start_hidden()
        probs = []
        for data in train_data:
            # data, label, hands= data["images"].to(device), data["label"].to(device), data["hands"].to(device)
            data, label = data["images"].to(device), data["label"].to(device)

            label = label.unsqueeze(1).repeat(1, data.size(1))
            
            model.zero_grad()
            for seq in range(max_sequence//sequence):
                b = seq*sequence
                e = (seq+1)*(sequence) if seq == 0 else max_sequence
                seq_data = data[:,b:e,:,:,:]
                # seq_data = data[:,b:e]
                seq_label = label[:,b:e]
                # seq_hands = hands[:,b:e,:,:,:]
                seq_label = seq_label.contiguous().view(-1)
                out = model(seq_data)
                # hidden = ([h.data for h in hidden])

                # if model.module.type == bl.ModelType.MC_DROP or  model.module.type == bl.ModelType.DET:
                if model.type == bl.ModelType.MC_DROP or  model.type == bl.ModelType.DET:

                    l1 = None
                    for p in model.parameters():
                        l1 = p.norm(1) if l1 is None else l1 + p.norm(1)
                
                    NLL, _ = model.get_loss(out, seq_label)
                    # proper scaling for sequence loss
                    loss = NLL   + params["weight_decay"] * l1
                    #backpropagate error
                else:
                    loss,_ = model.module.get_loss(out, seq_label)
                    # NLL, KL = model.module.get_loss(out, seq_label)
                    # # proper scaling for sequence loss
                    # NLL_term = NLL / C
                    # # proper scaling for sequence loss
                    # kl_term = scale*(KL / B*C)
                    # #Composed loss
                    # loss = NLL_term + kl_term 
                
                loss.backward()
                #clips gradients before update weights
                nn.utils.clip_grad_norm_(model.parameters(), clip)
                #update weights
                optimizers[0].step()
                optimizers[1].step()

                model.module.start_hidden()
                # hidden = None
                out = out.view(-1, sequence,out.size(1))[:,-1,:]
                seq_label = seq_label.view(-1, sequence)[:,-1]
                # out = out.view(-1, sequence,out.size(1))#spotting
                # seq_label = seq_label.view(-1, sequence) #spotting
                prob, preds = torch.max(out, 1)
                probs.append(prob.cpu().detach().numpy())
                total += out.size(0)
                running_loss += loss.item() * out.size(0)
                running_kl += loss.item() * out.size(0)
                running_corrects += torch.sum(preds == seq_label.data).double()
            
        
        acc_train = running_corrects/float(total)
        accuracies.append(acc_train )
        if epoch > 9 and acc_train<0.4:break

        if acc_train  > 0.7:
            schedulers[0].step()
            schedulers[1].step()
        
        if acc_train > 0.9:
            acc_val = predict_DLSTM(model,val_data,params,[])
            

            if acc_val > best_acc_val: 
                best_acc_val = acc_val
                dict_save = {"acc":best_acc_val,"params":params, "model":model.module.state_dict()}
                torch.save(dict_save, "dynamic_star_rgb_RNN_val.pth")


            
        if epoch % 10 == 0:
           myprint("Epoch = {}  Train ({:.2f}%/{:.2f})  Val ({:.2f}%/{:.2f}%)".format(epoch,acc_train.cpu().numpy()*100, loss.item(), acc_val*100,best_acc_val*100), telegram=True)

        
        #early stop conditions
        if acc_val > 0.97:break
        if len(accuracies)  > 3: 
            del accuracies[0]
            mean = sum(accuracies)/float(len(accuracies))
            if  mean > 0.99: break
    
    myprint("Epoch = {}  Train ({:.2f}%/{:.2f})  Val ({:.2f}%/{:.2f}%)".format(epoch,acc_train.cpu().numpy()*100, loss.item(), acc_val*100,best_acc_val*100), telegram=True)
    return model



def fixed_sequece_size(seq, max):
    sequences = np.zeros((max,len(seq[0])))-1
    values  = seq[:max] if len(seq) >= max else seq
    sequences[:len(values)] = values
    return sequences

def get_observation(seq, t, max):
    sequences = np.zeros((max,len(seq[0])))-1
    if t >=max: t=max
    values  = seq[:t] if len(seq) >= t else seq
    sequences[:len(values)] = values
    return sequences


def viterbi_approximation(model, observation):
        A = model.transmat_
        m = model.means_
        cv = model.covars_
       
        states = model.predict(observation)
        # print(S.shape)
        B = model.predict_proba(observation)
        v = model.startprob_.tolist()[states[0]]
        v *= multivariate_normal(m[states[0]],cv[states[0]]).pdf(observation[0])
        for i, (o, s_i,s_j) in enumerate(zip(observation[1:], states[:-1],states[1:])):
            b = multivariate_normal.pdf(o,m[s_j],cv[s_j])
            v *= A[s_i,s_j]*b
        return v

def softmax(x, axis= 1):
    """Compute softmax values for each sets of scores in x."""
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def HMM_train(train, params):
    ste, seq = params["states"], params["max_seq"]
   
    models = [hmm.GaussianHMM(n_components=params["states"],  verbose=False, tol = 0.01, n_iter = 30) for _ in range(params["num_classes"])]
    # for model in models:
    #     model.transmat_ = np.ones((params["states"],params["states"]))/params["states"]

    sequencies = [[] for _ in range(params["num_classes"])]
    lengths = [[] for _ in range(params["num_classes"])]

    for d,l in train:
        sequencies[l] += fixed_sequece_size(d,params["max_seq"]).tolist()
        lengths[l] += [params["max_seq"]]#[e-b]

    sequencies = np.array(sequencies)
    lengths = np.array(lengths)
    for i, (s, l) in enumerate(zip(sequencies,lengths)):
        models[i] = models[i].fit(s,l)
    return models

def HMM_test(models, test,params, results):
    predictions = np.zeros((len(test),params["num_classes"]))
    
    labels = []
    for n, (d,l,i) in enumerate(test):
        video_preds  = []
        labels.append(l)
        for t in range(1,len(d)):
            obs_pred = np.zeros(params["num_classes"])
            observation = get_observation(d,t,params["max_seq"]) #d[:t]
            for i,model in enumerate(models):
                score = model.score(observation)
                obs_pred[i] = score #viterbi_approximation(model, observation)
            obs_pred = obs_pred - obs_pred.min() #(obs_pred - obs_pred.min())/(obs_pred.max() - obs_pred.min()+1e-18)
            obs_pred[obs_pred==0] = 1e-6
            obs_pred = obs_pred/obs_pred.sum()
            # obs_pred = np.log(obs_pred)

            # # obs_pred = (np.exp(obs_pred)/np.exp(obs_pred).sum())
            # if video_preds:
            #     video_preds.append(video_preds[-1]+obs_pred)
            # else:
            #     video_preds.append(obs_pred)
            
            video_preds.append(obs_pred)

        video_preds = np.array(video_preds)
        results.append({"pred":np.argmax(video_preds,1)[-1], "label":l,  "probs":video_preds, "interval":i})
        predictions[n] = video_preds[-1]
        # print(np.exp(video_preds[-1]))
    labels = np.array(labels)
    acc = (np.argmax(predictions,1) == labels).sum()/len(labels)
   
    return  acc


def NB_train(train, params):
    model = GaussianNB()
    labels, sequencies = [], []
    for d,l in train:
        labels.append(l)
        sequencies.append(fixed_sequece_size(d,params["max_seq"]).reshape(-1))

    sequencies = np.array(sequencies)
    labels = np.array(labels)
    model= model.fit(sequencies,labels)
    return model


def NB_test(model, test, params, results):
    predictions = np.zeros((len(test),params["num_classes"]))
    labels = []
    for n, (d,l,i) in enumerate(test):
        video_preds  = []
        labels.append(l)
        for t in range(1,len(d)):
            observation = get_observation(d,t,params["max_seq"]).reshape(1,-1) #d[:t]
            obs_pred = model.predict_proba(observation)[0]
            # obs_pred[obs_pred==0] = 1e-6
            # obs_pred = obs_pred/obs_pred.sum()
            # obs_pred = np.log(obs_pred)

            # obs_pred = (np.exp(obs_pred)/np.exp(obs_pred).sum())
            # if video_preds:
            #     video_preds.append(video_preds[-1]+obs_pred)
            # else:
            #     video_preds.append(obs_pred)

            video_preds.append(obs_pred)
        video_preds = np.array(video_preds)
        results.append({"pred":np.argmax(video_preds,1)[-1], "label":l,  "probs":video_preds, "interval":i})
        predictions[n] = video_preds[-1]
        # print(np.exp(video_preds[-1]))
    labels = np.array(labels)
    acc = (np.argmax(predictions,1) == labels).sum()/len(labels)
    return  acc


def SVM_train(train, params):
    model = LinearSVC(loss=params["loss"],  
                      C=params["c"], 
                      multi_class=params["multi_class"], 
                      max_iter=20000,
                      tol = 1e-1,
                      verbose=False,
                      fit_intercept = False)
    labels, sequencies = [], []
    for d,l in train:
        labels.append(l)
        sequencies.append(fixed_sequece_size(d,params["max_seq"]).reshape(-1))

    sequencies = np.array(sequencies)
    labels = np.array(labels)
    model= model.fit(sequencies,labels)
    return model




def SVM_test(model, test, params, results):
    predictions = np.zeros((len(test),params["num_classes"]))
    labels = []
    for n, (d,l,i) in enumerate(test):
        video_preds  = []
        labels.append(l)
        for t in range(1,len(d)):
            observation = get_observation(d,t,params["max_seq"]).reshape(1,-1) #d[:t]
            obs_pred = model.decision_function(observation)[0]
            # print(obs_pred)
            # obs_pred[obs_pred==0] = 1e-6
            # obs_pred = obs_pred/obs_pred.sum()
            # obs_pred = np.log(obs_pred)

            obs_pred = (np.exp(obs_pred)/np.exp(obs_pred).sum())
            # if video_preds:
            #     video_preds.append(video_preds[-1]+obs_pred)
            # else:
            #     video_preds.append(obs_pred)
           
            video_preds.append(obs_pred)
        video_preds = np.array(video_preds)
        results.append({"pred":np.argmax(video_preds,1)[-1], "label":l,  "probs":video_preds, "interval":i})
        predictions[n] = video_preds[-1]
        # print(np.exp(video_preds[-1]))
    labels = np.array(labels)
    acc = (np.argmax(predictions,1) == labels).sum()/len(labels)
    return  acc


def NSVM_train(train, params):
    model = NuSVC(   nu=0.5,  
                     kernel=params["kernel"], 
                     degree=params["degree"], 
                     decision_function_shape = "ovr",
                     max_iter=200000,
                     tol = 1e-3,
                     gamma = "scale",
                     verbose=False)
    labels, sequencies = [], []
    for d,l in train:
        labels.append(l)
        sequencies.append(fixed_sequece_size(d,params["max_seq"]).reshape(-1))

    sequencies = np.array(sequencies)
    labels = np.array(labels)
    model= model.fit(sequencies,labels)
    return model




def KNN_train(train, params):
    model = KnnDtw()
    labels, sequencies = [], []
    for d,l in train:
        labels.append(l)
        sequencies.append(fixed_sequece_size(d,params["max_seq"]).reshape(-1))

    sequencies = np.array(sequencies)
    labels = np.array(labels)

    model = model.fit(sequencies,labels)
    return model




def KNN_test(model, test, params, results):
    predictions = np.zeros(len(test))
    labels = []
    for n, (d,l) in enumerate(test):
        video_preds  = []
        labels.append(l)
        # for t in range(1,len(d)):
        observation = get_observation(d,len(d)-1,params["max_seq"]).reshape(1,-1) #d[:t]
        obs_pred, proba = model.predict(observation)
        # print(f"{obs_pred} {proba} {l}")
        # obs_pred[obs_pred==0] = 1e-6
        # obs_pred = obs_pred/obs_pred.sum()
        # obs_pred = np.log(obs_pred)

        # obs_pred = (np.exp(obs_pred)/np.exp(obs_pred).sum())
        # if video_preds:
        #     video_preds.append(video_preds[-1]+obs_pred)
        # else:
        video_preds.append(proba[0])

    #         video_preds.append(obs_pred)
    #     video_preds = np.array(video_preds)
    #     results.append({"pred":np.argmax(video_preds,1)[-1], "label":l,  "probs":video_preds, "interval":[0,1]})
        predictions[n] = obs_pred[0]
    #     # print(np.exp(video_preds[-1]))
    # labels = np.array(labels)
    acc = (np.argmax(predictions) == labels).sum()/len(labels)
    print(acc*100)
    # return  acc

def CONV_train(train, params):
    model = ConvModel(params)
    criterion = nn.NLLLoss()
    optimizer = Adam(model.parameters(), lr=params["lr"])
    batch_size = params["batch"]
    indices = list(range(len(train)))
    steps = len(indices)//batch_size
    sequencies = []
    labels = []
    for d,l in train:
        labels.append(l)
        sequencies.append(np.transpose(fixed_sequece_size(d,params["max_seq"])))

    sequencies = np.array(sequencies)
    labels = np.array(labels)
    for epoch in range(50):
        np.random.shuffle(indices)
        corrects = 0
        for step in range(steps):
            model.zero_grad()
            begin = step*batch_size
            end = begin+batch_size
            if step == steps-1: end = len(indices)
            batch = torch.from_numpy(sequencies[indices[begin:end]]).float()
            label_batch = torch.from_numpy(labels[indices[begin:end]]).long()
            out = model(batch)

            l2 = None
            for p in model.parameters():
                l2 = (p*p).sum() if l2 is None else l2 + (p*p).sum()

            loss = criterion(out, label_batch) + params["weight_decay"]*l2
            loss.backward()
            optimizer.step()
            corrects += (out.argmax(1) == label_batch).sum().data.numpy()

        acc = corrects/len(labels)
        if acc >0.99:
            break
    return model
            
def CONV_test(model, test, params, results):
    predictions = np.zeros((len(test),params["num_classes"]))
    labels = []
    for n, (d,l,i) in enumerate(test):
        video_preds  = []
        labels.append(l)
        for t in range(1,len(d)):
            observation = torch.from_numpy(np.transpose(get_observation(d,t,params["max_seq"]))).unsqueeze(0).float()
            obs_pred = model(observation).exp().detach().numpy()[0]
            video_preds.append(obs_pred)
        video_preds = np.array(video_preds)
        results.append({"pred":np.argmax(video_preds,1)[-1], "label":l,  "probs":video_preds, "interval":i})
        predictions[n] = video_preds[-1]
        # print(np.exp(video_preds[-1]))
    labels = np.array(labels)
    acc = (np.argmax(predictions,1) == labels).sum()/len(labels)
    return  acc


def RNN_train(train, params):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RNNModel(params)
    model = model.to(device)
    model.train()
    criterion = nn.NLLLoss()
    optimizer = Adam(model.parameters(), lr=params["lr"])
    batch_size = params["batch"]
    indices = list(range(len(train)))
    steps = len(indices)//batch_size
    sequencies = []
    labels = []
    for d,l in train:
        labels.append(np.zeros(params["max_seq"])+l)
        sequencies.append(fixed_sequece_size(d,params["max_seq"]))

    sequencies = np.array(sequencies)
    labels = np.array(labels)

    seq_len = params["seq_len"] if params["max_seq"] > params["seq_len"] else params["max_seq"]
    max = params["max_seq"]
    seq_steps = params["max_seq"] // seq_len
    # print(f"{seq_len} {max} {seq_steps}")
    for epoch in range(50):
        np.random.shuffle(indices)
        corrects = 0
        losses = 0
        for step in range(steps):
            begin = step*batch_size
            end = begin+batch_size
            # if step == steps-1: end = len(indices)
           
            batch = torch.from_numpy(sequencies[indices[begin:end]]).float()
            label_batch = torch.from_numpy(labels[indices[begin:end]]).long()

            model.hx= None
            
            for s in range(seq_steps):
                bs = s*seq_len
                es = bs+seq_len
                if s == seq_steps-1: es = params["max_seq"]
                batch_seq = batch[:,bs:es].to(device)
                seq_label = label_batch[:,bs:es].contiguous().view(-1).to(device)
                out = model(batch_seq)
                l2 = None
                for p in model.parameters():
                    l2 = p.norm(2) if l2 is None else l2 + p.norm(2)
                out = out.view(-1,params["num_classes"])
                loss = criterion(out, seq_label) + 0.0001*l2
                optimizer.zero_grad()
                loss.backward()
                
                optimizer.step()
                losses += loss.item()
    return model
            
def RNN_test(model, test, params, results):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    predictions = np.zeros((len(test),params["num_classes"]))
    labels = []
    for n, (d,l,i) in enumerate(test):
        video_preds  = []
        labels.append(l)
        sequence = torch.from_numpy(fixed_sequece_size(d,params["max_seq"])).unsqueeze(0).float().to(device)
        # print(f"--------------- {sequence.shape}")
        model.hx = None
        video_preds = model(sequence).exp().cpu().detach().numpy()[0]
        results.append({"pred":np.argmax(video_preds,1)[-1], "label":l,  "probs":video_preds, "interval":[0,1]})
        predictions[n] = video_preds[-1]
    labels = np.array(labels)
    acc = (np.argmax(predictions,1) == labels).sum()/len(labels)
    # print(acc*100)
    return  acc


def MLP_train(train, params):
    model = MLPClassifier(hidden_layer_sizes = params["hidden_dim"], 
                            learning_rate = "adaptive", 
                            learning_rate_init = params["lr"],
                            max_iter= 10000)
    labels, sequencies = [], []
    for d,l in train:
        labels.append(l)
        sequencies.append(fixed_sequece_size(d,params["max_seq"]).reshape(-1))

    sequencies = np.array(sequencies)
    labels = np.array(labels)
    model= model.fit(sequencies,labels)
    return model



def MLP_test(model, test, params, results):
    predictions = np.zeros((len(test),params["num_classes"]))
    labels = []
    for n, (d,l,i) in enumerate(test):
        video_preds  = []
        labels.append(l)
        for t in range(1,len(d)):
            
            observation = get_observation(d,t,params["max_seq"]).reshape(1,-1) #d[:t]
            obs_pred = model.predict_proba(observation)[0]
            video_preds.append(obs_pred)
        video_preds = np.array(video_preds)
        results.append({"pred":np.argmax(video_preds,1)[-1], "label":l,  "probs":video_preds, "interval":i})
        predictions[n] = video_preds[-1]
    labels = np.array(labels)
    acc = (np.argmax(predictions,1) == labels).sum()/len(labels)
    return  acc

def load_model( path = "dynamic_star_rgb_RNN_val.pth"):
    dict_save = torch.load(path)
    params = dict_save["params"]
    myprint( "Loaded acc = {}".format(dict_save["acc"]),True)
    # model = DStarRGBSpottingModel(
    #             output_size =  params["num_classes"], 
    #             hidden_dim =  params["hidden_dim"], 
    #             n_layers =  params["n_layers"],  
    #             mode =  "DET",
    #             dropout = params["fc_drop"], 
    # )
    model = DStarRGBModel(
                output_size =  params["num_classes"], 
                hidden_dim =  params["hidden_dim"], 
                n_layers =  params["n_layers"],  
                mode =  "DET",
                dropout = params["fc_drop"], 
    )
    # model = DStarRGBHandSpottingModel(
    #             output_size =  params["num_classes"], 
    #             hidden_dim =  params["hidden_dim"], 
    #             n_layers =  params["n_layers"],  
    #             mode =  "DET",
    #             dropout = params["fc_drop"], 
    # )
    # model = nn.DataParallel(model)
    model.load_state_dict(dict_save["model"])
    return model,params

def retraining_best():
    start = timer()
    model,params = load_model()
    params["lr"] = [5e-5,5e-4]
    params["epoch"] = 50
    params["batch_size"] = 96
    params["max_seq"] = (16,32)
    myprint( "Hyperparameters = {}".format(params))
    loaders = create_datasets(num_workers=30, batch_size=params["batch_size"], max_size =params["max_seq"][1], alpha =params["alpha"], window =params["window"] )

    model = train_LSTM(model,loaders["train"],loaders["val"],params)
    
    acc = predict_DLSTM(model.module,loaders["test"],params,[])
    myprint( "Teste acc = {:.3f}".format(acc*100))
    dict_save = {"acc":acc,"params":params, "model":model.state_dict()}
    name = "dynamic_star_rgb_{}.pth".format(int(acc*10000))
    torch.save(dict_save, name)
    
    model,params = load_model()

    acc = predict_DLSTM(model,loaders["test"],params,[])
    myprint( "Teste acc = {:.3f}".format(acc*100))
    dict_save = {"acc":acc,"params":params, "model":model.state_dict()}
    name = "dynamic_star_rgb_{}.pth".format(int(acc*10000))
    torch.save(dict_save, name)

    end = timer()
    elapsed = end - start
    h = int(elapsed//3600)
    m = int((elapsed - h*3600)//60)
    s = int(elapsed - h*3600 - m*60)

    myprint("Elapsed time {}:{}:{}".format(str(h).zfill(2),str(m).zfill(2),str(s).zfill(2)))

def predict_model_save():
    model,params = load_model()
    myprint(params)
    loaders =  create_datasets(num_workers=30, batch_size=params["batch_size"], max_size =params["max_seq"][1], alpha =params["alpha"], window =params["window"] )
    
    acc = predict_DLSTM(model,loaders["test"],params,[])
    print(acc)
    dict_save = {"acc":acc,"params":params, "model":model.state_dict()}
    name = "dynamic_star_rgb_{}.pth".format(int(acc*10000))
    torch.save(dict_save, name)   
    myprint(acc)




def telegram_bot_sendtext(bot_message):
    try:   
        bot_token = '754818149:AAGlBDM_u5xovmeM8liGh3qpIQHq8IFNIMc'
        bot_chatID = '887550379'
        send_text = "https://api.telegram.org/bot{}/sendMessage?chat_id={}&text={}".format(bot_token,bot_chatID,bot_message)
        response = requests.get(send_text)
        return response.json()
    except:
        return None

def myprint(message, telegram = True):
    print(message)
    if telegram:telegram_bot_sendtext(message)
            

if __name__ == "__main__":
    # myprint("Starting...")
    # retraining_best()
    # predict_model_save()
    # params = {
    #         "num_classes":20,
    #         "gpu":0,
    #         "epoch":70,
    #         "batch_size":96,
    #         "max_seq": (32,32),
    #         "n_layers":2,
    #         "hidden_dim":1024,
    #         "fc_drop":0.1,
    #         "clip":3.0,
    #         "lr":[1e-4,1e-3],
    #         "weight_decay":1e-5,
    #         }
    # loaders = create_datasets(num_workers=20, batch_size=params["batch_size"], max_size =params["max_size"] )

    model,params = load_model("dynamic_star_rgb_9527.pth")
    # files = glob.glob("/notebooks/datasets/Montalbano/*/numpy_files/star_rgb*.npz")
    # for file in files:
    #     save_bottleneack(model,file)
    # params["lr"] = [1e-5,1e-4]
    # params["weight_decay"] = 3e-5
    # loaders = create_datasets(num_workers=20, batch_size=params["batch_size"], max_size =params["max_seq"][1] )
    # model = train_LSTM(model,loaders["train"],loaders["val"],params)
                
    # dict_save = {"params":params, "model":model.module.state_dict()}
    # torch.save(dict_save, "dynamic_star_rgb.pth")
    results = []
    test_data = DStarRGBDataset(dataset="test", max_size = 32, alpha = 0.6,
                            transform=DataTrasformation(output_size=(110,120), data_aug = False))
    test_data = DataLoader(test_data, 
                                batch_size=30, pin_memory=True, shuffle=False, 
                                num_workers=20, drop_last=False)
    acc = predict_DLSTM(model,test_data,params,results)
    print(acc)
    with open('results_starRGB_LSTM_{}.pkl'.format(int(acc*10000)), 'wb') as file:
        pickle.dump(results, file, protocol=pickle.HIGHEST_PROTOCOL)

    # myprint( "Teste acc = {}".format(acc),True)
    
    # dict_save = {"acc":acc,"params":params, "model":model.module.state_dict()}
    # torch.save(dict_save, "dynamic_star_rgb.pth")