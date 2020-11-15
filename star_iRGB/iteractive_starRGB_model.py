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
import time
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
import iteractive_starRGB_dataset as starRGB_dataset
import utils
torch.manual_seed(30)
np.random.seed(30)
random.seed(30) 



class IterStarRGBModel(nn.Module):
    def __init__(self,  output_size, hidden_dim,n_layers, mode = "mc", dropout = 0.5, logstd1 = -1, logstd2 = -2, pi = 0.5 ):
        super(IterStarRGBModel, self).__init__()
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
        self.conv1d = nn.Conv1d(1, 1, 3, stride=2)
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
                                    
        self.dropout = dropout
        # linear and sigmoid layers
        self.fc = bl.Linear(self.hidden_dim, self.output_size, **last_linear_args)
        self.dp_training = True
        self._baysian_layers = []
        self.hidden = None
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device('cpu')

    def start_hidden(self):
        self.hidden = None

    @staticmethod
    def load_model(path ):
        dict_save = torch.load(path)
        params = dict_save["params"]
        utils.log( "Loaded acc = {}".format(dict_save["acc"]),True)
        model = IterStarRGBModel(
                    output_size =  params["num_classes"], 
                    hidden_dim =  params["hidden_dim"], 
                    n_layers =  params["n_layers"],  
                    mode =  "DET",
                    dropout = params["fc_drop"], 
        )

        model.load_state_dict(dict_save["model"])
        return model,params
            

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
        x = x.squeeze(2).squeeze(2).unsqueeze(1)
        x = F.relu(self.conv1d(x)).squeeze()
        x = x.view(size[0],size[1],-1)

        out, hidden_out = self.lstm(x, self.hidden)
        self.hidden = ([h.data for h in hidden_out])

        out = self.fc(out) 
        out = out.contiguous().view(-1, out.size(-1))
        return out

    def set_dropout(self, value, training=True):
        self.dropout = value
        for layer in self.get_baysian_layers(): layer.dropout = value


    def get_baysian_layers(self):
        if not self._baysian_layers:
            self._baysian_layers = [module for module in self.modules() if isinstance(module,bl.BaseLayer)]
        return self._baysian_layers


    def get_loss(self, output, targets):
        NLL = self.loss_fn(F.log_softmax(output,1), targets)
        return NLL
    
    def predict(self, test_data, results = None):
        model = self.to(self.device)
        model.eval()
        running_corrects = 0
        total = 0
        with torch.no_grad():
            for data in test_data:
                data, label = data["images"].to(self.device), data["label"]
                if len(data.shape) < 5: data = data.unsqueeze(0)
                self.start_hidden()
                out = model(data)
                out = out.cpu()
                probs = F.log_softmax(out,1).exp().detach().numpy()
                if self.output_size > 2:
                    probs = probs.reshape((data.size(0),data.size(1),-1))[:,-1,:] #take the last prediction from each sequence
                else:
                    probs = probs.reshape((data.size(0),data.size(1),-1))#spotting
                label = label.reshape(-1).numpy()
                pred = np.argmax(probs, 1)
                running_corrects += np.sum(pred == label)
                if results is not None:
                    results.append({"pred":pred, "label":label, "probs":probs})
                total+= len(pred)
                
        return running_corrects/total
        

    def fit(self,train_data, val_data, params):
        model = nn.DataParallel(self)
        model = model.to(self.device)
        model.train()
        #Hyperparameters
        epochs = params["epoch"]
        batch = params["batch_size"]
        sequence = params["max_seq"][0] 
        max_sequence = params["max_seq"][1] 
        clip = params["clip"]
        lr=params["lr"]
        ft, clf = self.get_parameters()
        optimizers = [ optim.Adam(ft, lr=lr[0]), optim.Adam(clf, lr=lr[1])]
        schedulers = [StepLR(optimizers[0], step_size = 1, gamma = 0.99),
                    StepLR(optimizers[1], step_size = 1, gamma = 0.99) ]

        acc_val = 0
        best_acc_val = 0
        accuracies = []
        for epoch in range(epochs):
            running_loss = 0
            running_corrects = 0 
            total = 1e-18
            self.start_hidden()
            for data in train_data:
                data, label = data["images"].to(self.device), data["label"].to(self.device)
                label = label.repeat(1, data.size(1))
                model.zero_grad()
                for seq in range(max_sequence//sequence):
                    b = seq*sequence
                    e = (seq+1)*(sequence) if seq == 0 else max_sequence
                    seq_data = data[:,b:e,:,:,:]
                    seq_label = label[:,b:e]
                    seq_label = seq_label.contiguous().view(-1)
                    out = model(seq_data)
                    l1 = None
                    for p in model.parameters():
                        l1 = p.norm(1) if l1 is None else l1 + p.norm(1)
                
                    nll = self.get_loss(out, seq_label)
                    loss = nll   + params["weight_decay"] * l1                    
                    loss.backward()
                    #clips gradients before update weights
                    nn.utils.clip_grad_norm_(model.parameters(), clip)
                    
                    #update weights
                    optimizers[0].step()
                    optimizers[1].step()

                    # hidden = None
                    self.start_hidden()

                    if self.output_size > 2: #take the last prediction from each sequence
                        out = out.view(-1, sequence,out.size(1))[:,-1,:]
                        seq_label = seq_label.view(-1, sequence)[:,-1] 
                    else: #spotting
                        out = out.view(-1, sequence,out.size(1))
                        seq_label = seq_label.view(-1, sequence) 

                    prob, preds = torch.max(out, 1)
                    total += out.size(0)
                    running_loss += loss.item() * out.size(0)
                    running_corrects += torch.sum(preds == seq_label.data).double()
            
            acc_train = running_corrects/float(total)
            accuracies.append(acc_train )
            if epoch > 9 and acc_train<0.4:break

            if acc_train  > 0.7:
                schedulers[0].step()
                schedulers[1].step()
            
            if acc_train > 0.9:
                acc_val = self.predict(val_data)
                

                if acc_val > best_acc_val: 
                    best_acc_val = acc_val
                    dict_save = {"acc":best_acc_val,"params":params, "model":model.module.state_dict()}
                    torch.save(dict_save, "{}_val.pth".format(params["name"]))


                
            if epoch % 10 == 0:
                utils.log("Epoch = {}  Train ({:.2f}%/{:.2f})  Val ({:.2f}%/{:.2f}%)".format(epoch,acc_train.cpu().numpy()*100, loss.item(), acc_val*100,best_acc_val*100), telegram=True)

            
            #early stop conditions
            if acc_val > 0.97:break
            if len(accuracies)  > 3: 
                del accuracies[0]
                mean = sum(accuracies)/float(len(accuracies))
                if  mean > 0.99: break
        utils.log("Epoch = {}  Train ({:.2f}%/{:.2f})  Val ({:.2f}%/{:.2f}%)".format(epoch,acc_train.cpu().numpy()*100, loss.item(), acc_val*100,best_acc_val*100), telegram=True)

def do_synthetic_teste():
    samples = 10000
    images = torch.from_numpy(np.zeros((1,samples,3, 110,120))).float()
    start = time.time()
    model = IterStarRGBModel(
            output_size =   20, 
            hidden_dim =  1024, 
            n_layers =  2,  
            mode =  "DET",
            dropout = 0.1,
            )
    
    model = model.to(model.device)
    model.eval()
    for i in range(samples):
        image = images[:,i].unsqueeze(1)
        # print(image.shape)
        model(image.to(model.device))
    end = time.time() - start
    print(end/samples)
    

if __name__ == "__main__":
    do_syntethic_teste()
    model,params = IterStarRGBModel.load_model("dynamic_star_rgb_LSTM_9488.pth")
    utils.log( "Hyperparameters = {}".format(utils.__get_params(params)))
    loader = starRGB_dataset.create_datasets(num_workers=20, batch_size=params["batch_size"], 
                                            alpha = params["alpha"], window = params["window"], 
                                             max_size =params["max_seq"][1] )

    model.fit(loader["train"], loader["val"], params)
    model,params = IterStarRGBModel.load_model("{}_val.pth".format(params["name"]))
    results = []
    acc = model.predict(loader["test"],results)
    with open('results_{}_{}.pkl'.format(params["name"], int(acc*10000)), 'wb') as file:
            pickle.dump(results, file, protocol=pickle.HIGHEST_PROTOCOL)
    dict_save = {"acc":acc,"params":params, "model":model.state_dict()}
    torch.save(dict_save, "{}_{}.pth".format(params["name"], int(acc*10000) ))
    utils.log("Test acc = {:.2f}%".format(acc*100))