

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

import utils
from hyperopt import hp
deterministic_experiment(30)

def deterministic_experiment(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

class TimeDistributed(nn.Module):
    def __init__(self):
        super(TimeDistributed, self).__init__()
        
    def forward(self,module, x):
        if type(x) == list: 
            seq_size = x[0].size(1)
        else:
            seq_size = x.size(1)
        outputs = []
        for t in range(seq_size):
            if type(x) == list: 
                x_t = [x_i[:,t] for x_i in x]
            else:
                x_t = x[:,t]
            out = module(x_t)
            if len(out.shape) < 3: out = out.unsqueeze(1)
            outputs.append(out)
        return torch.cat(outputs,1)

class SoftAttention(nn.Module):
    def __init__(self,  input_size = 512, hidden_size = 128, type = "DET", dropout  = 0 ):
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
             "dropout":dropout
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
        return x.squeeze(2).squeeze(2).unsqueeze(1)

class IterStarRGBHandModel(nn.Module):
    def __init__(self,  output_size, hidden_dim,n_layers, mode = "mc", dropout = 0.5, logstd1 = -1, logstd2 = -2, pi = 0.5 ):
        super(IterStarRGBHandModel, self).__init__()
        #type = bl.ModelType.VAR_DROP_B_ADAP
        mode = mode.lower()
        if mode == "mc": self.type =  bl.ModelType.MC_DROP
        elif mode == "vd": self.type = bl.ModelType.VAR_DROP_B_ADAP
        elif mode == "bbb": self.type = bl.ModelType.BBB
        else: self.type = bl.ModelType.DET
        
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
             "type":bl.ModelType.DET,
             "dropout":0
             }

        resnet_mov = models.resnet50(pretrained=True)
        self.mov= nn.Sequential(*(list(resnet_mov.children())[:-1]),SqueezeExtractor(), nn.Conv1d(1, 1, 3, stride=2),nn.ReLU())
        resnet_hand = models.resnet50(pretrained=True)
        self.hand= nn.Sequential(*(list(resnet_hand.children())[:-1]), SqueezeExtractor(), nn.Conv1d(1, 1, 3, stride=2),nn.ReLU())

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
                                    

        self.dropout = dropout
        self.fc = bl.Linear(self.hidden_dim, self.output_size, **last_linear_args)
        self.dp_training = True
        self._baysian_layers = []
        self.hidden = None
        self.td = TimeDistributed()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.device = torch.device( 'cpu')
        self.frozen = False


    def get_parameters(self):
        extractor = list(self.mov.parameters())
        extractor += self.hand.parameters()
        classifier = list(self.soft.parameters())
        classifier += self.lstm.parameters()
        classifier += self.fc.parameters()
        return extractor, classifier
        
    def start_hidden(self):
        self.hidden = None


    def forward(self, m, h,  mc_simulation = None):
        B,N,C,H,W = m.size()
        m = m.contiguous().view(B*N,C,H,W)
        m = self.mov(m)
        B,N,C,H,W = h.size()
        h = h.contiguous().view(B*N,C,H,W)
        h = self.hand(h)
        x = self.soft([m,h])
        x = x.view(B,N,-1)
        if mc_simulation:
            x = x.repeat(mc_simulation,1,1)
        out, hidden_out = self.lstm(x, self.hidden)
        self.hidden = ([h.data for h in hidden_out])
        out = self.td(self.fc,out) 
        out = out.contiguous().view(-1, out.size(-1))
        return out


    def freeze_CNN(self, freeze = False):
        print("Freezing CNN" if freeze else "Unfreezing CNN" )
        for p in self.mov.parameters():
            p.requires_grad = not freeze
        for p in self.hand.parameters():
            p.requires_grad = not freeze
        self.frozen = freeze

    @staticmethod
    def load_model(path, mode = "DET" ):
        dict_save = torch.load(path)
        params = dict_save["params"]
        utils.log( "Loaded acc = {}".format(dict_save["acc"]),True)
        params["fc_drop"] = 0.2
        model = IterStarRGBHandModel(
                    output_size =  params["num_classes"], 
                    hidden_dim =  params["hidden_dim"], 
                    n_layers =  params["n_layers"],  
                    mode =  mode,
                    dropout = params["fc_drop"], 
        )

        model.load_state_dict(dict_save["model"])
        return model,params

    def set_dropout(self, value, training=True):
        for layer in self.get_bayesian_layers(): 
            layer.dropout = value


    def get_bayesian_layers(self):
        if not self._baysian_layers:
            self._baysian_layers = [module for module in self.modules() if ((isinstance(module,bl.BaseLayer) or isinstance(module,bl.BaseRNNLayer))  and module._type !=  bl.ModelType.DET) ]
        return self._baysian_layers


    def get_loss(self, output, targets):
        kl = sum([layer.get_kl() for layer in self.get_bayesian_layers()])
        NLL = self.loss_fn(F.log_softmax(output,1), targets)
        return NLL + 0.1*kl
    
    def predict(self, test_data, results = None):
        model = self.to(self.device)
        model.eval()
        mc_samples = None
        if self.type != bl.ModelType.DET:
            print("MC Samples activated...")
            mc_samples = 20
            self.set_dropout(0.30)

        running_corrects = 0
        total = 0
        complete_probs = None
        with torch.no_grad():
            for i, data in enumerate(test_data):
                data,label, hands = data["images"].to(self.device), data["label"].numpy(), data["hands"].to(self.device)
                if len(data.shape) < 5: data = data.unsqueeze(0)
                if len(hands.shape) < 5: hands = hands.unsqueeze(0)
                self.start_hidden()
                out = model(data,hands, mc_simulation = mc_samples)
                out = out.cpu()
                
                if self.output_size > 2: #take the last prediction from each sequence
                    out = out.view(-1, data.size(1),out.size(1))
                    complete_probs = F.log_softmax(out,2).exp().detach().numpy()
                    out = out[:,-1,:]
                else:
                    probs = F.log_softmax(out,1).exp().detach().numpy()
                    complete_probs = probs

                
               
                if self.type != bl.ModelType.DET:
                    if self.output_size == 2:
                        out = out.view(-1, data.size(1),out.size(1))
                        probs = F.log_softmax(out,1).exp().detach().numpy()
                        probs = probs.mean(0)
                    else:
                        # print(probs.shape)
                        probs = F.log_softmax(out,1).exp().detach().numpy()
                        probs = probs.mean(0, keepdims=True)

                wc = self.soft.weights
                label = label.reshape(-1)
                pred = np.argmax(probs, 1)
                running_corrects += np.sum(pred == label)
                if results is not None:
                    results.append({"pred":pred, "label":label, "probs":complete_probs, "weights":wc})
                total+= len(pred)
        # if self.type != bl.ModelType.DET:
        #     self.set_dropout(self.dropout)
        return running_corrects/total
        

    def fit(self,train_data, val_data, params):
        # self.freeze_CNN(True)
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
                data,label, hands = data["images"].to(self.device), data["label"].to(self.device), data["hands"].to(self.device)
                if label.size(1) == 1:
                    label = label.repeat(1, data.size(1))
                model.zero_grad()
                for seq in range(max_sequence//sequence):
                    b = seq*sequence
                    e = (seq+1)*(sequence) if seq == 0 else max_sequence
                    seq_data = data[:,b:e,:,:,:]
                    seq_hands = hands[:,b:e,:,:,:]
                    seq_label = label[:,b:e]
                    seq_label = seq_label.contiguous().view(-1)
                    out = model(seq_data, seq_hands)
                    nll = self.get_loss(out, seq_label)
                    l1 = sum([p.norm(2) + p.norm(1) for p in model.parameters()]) #L1 + L2 regularizer
                    loss = nll   + params["weight_decay"] * l1                    
                    loss.backward()
                    #clips gradients before update weights
                    nn.utils.clip_grad_norm_(model.parameters(), clip)
                    
                    #update weights
                    if not self.frozen: 
                        optimizers[0].step()
                    optimizers[1].step()

                    # hidden = None
                    self.start_hidden()

                    if self.output_size > 2: #take the last prediction from each sequence
                        out = out.view(-1, sequence,out.size(1))[:,-1,:]
                        seq_label = seq_label.view(-1, sequence)[:,-1] 
                        
                    
                    prob, preds = torch.max(out, 1) 
                    total += out.size(0)
                    running_loss += loss.item() * out.size(0)
                    running_corrects += torch.sum(preds == seq_label.data).double()

                    
                   
            
            acc_train = running_corrects/float(total)
            accuracies.append(acc_train )
            if epoch > 9 and acc_train<0.4:break

            if acc_train  > 0.7:
                if not self.frozen: 
                    schedulers[0].step()
                schedulers[1].step()
            
            if acc_train > 0.9:
                acc_val = self.predict(val_data)
                model.train()
                

                if acc_val > best_acc_val: 
                    best_acc_val = acc_val
                    dict_save = {"acc":best_acc_val,"params":params, "model":model.module.state_dict()}
                    torch.save(dict_save, "{}_val.pth".format(params["name"]))


                
            if epoch % 10 == 0:
                utils.log("Epoch = {}  Train ({:.2f}%/{:.2f})  Val ({:.2f}%/{:.2f}%)".format(epoch,acc_train.cpu().numpy()*100, loss.item(), acc_val*100,best_acc_val*100), telegram=True)

            
            #early stop conditions
            if acc_val > 0.98:break
            if len(accuracies)  > 5: 
                del accuracies[0]
                mean = sum(accuracies)/float(len(accuracies))
                if  mean > 0.99: break
        utils.log("Epoch = {}  Train ({:.2f}%/{:.2f})  Val ({:.2f}%/{:.2f}%)".format(epoch,acc_train.cpu().numpy()*100, loss.item(), acc_val*100,best_acc_val*100), telegram=True)

def do_complete_test(path):
    model,params = IterStarRGBHandModel.load_model(path,"mc")
    test_data = starRGB_dataset.IterStarRGBHandDataset(dataset="test",alpha = params["alpha"], window = params["window"], 
                                                     max_size = None, spotting = model.output_size == 2,
                                                    transform=starRGB_dataset.DataTrasformation(output_size=(110,120), data_aug = False))
    utils.log( "Hyperparameters = {}".format(utils.__get_params(params)))
    results  = []
    acc = model.predict(test_data,results)
    dict_save = {"acc":acc,"params":params, "model":model.state_dict()}
    torch.save(dict_save, "{}_{}.pth".format(params["name"],int(acc*10000)))
    utils.log("Acc test = {:.2f}".format(acc*100))
    with open('results_{}_{}.pkl'.format(params["name"], int(acc*10000)), 'wb') as file:
            pickle.dump(results, file, protocol=pickle.HIGHEST_PROTOCOL)


def do_synthetic_teste():
    import time
    samples = 10000
    images = torch.from_numpy(np.zeros((1,samples,3, 110,120))).float()
    hands =  torch.from_numpy(np.zeros((1,samples,3, 40,80))).float()
    start = time.time()
    model = IterStarRGBHandModel(
            output_size =   20, 
            hidden_dim =  1024, 
            n_layers =  1,  
            mode =  "mc",
            dropout = 0.1,
            )
    
    model = model.to(model.device)
    model.eval()
    for i in range(samples):
        image = images[:,i].unsqueeze(1).to(model.device)
        hand  = hands[:,i].unsqueeze(1).to(model.device)
        # print(image.shape)
        model(image,hand, mc_simulation = 50)
    end = time.time() - start
    print(end/samples)


def do_retrain():
    model,params = IterStarRGBHandModel.load_model("dynamic_star_rgb_hand_SPT_LSTM_9354.pth","det")
    utils.log( "Hyperparameters = {}".format(utils.__get_params(params)))
    loader = starRGB_dataset.create_datasets(num_workers=25, batch_size=params["batch_size"], 
                                        alpha = params["alpha"], window = params["window"], 
                                            max_size =params["max_seq"][1], spotting = model.output_size == 2)
    

    model.fit(loader["train"], loader["val"], params)
    model,params = IterStarRGBHandModel.load_model("{}_val.pth".format(params["name"]),"mc")
    results = []
    model.set_dropout(0.2)
    acc = model.predict(loader["test"],results)
    with open('results_{}_{}.pkl'.format(params["name"], int(acc*10000)), 'wb') as file:
            pickle.dump(results, file, protocol=pickle.HIGHEST_PROTOCOL)
    dict_save = {"acc":acc,"params":params, "model":model.state_dict()}
    torch.save(dict_save, "{}_{}.pth".format(params["name"], int(acc*10000) ))
    utils.log("Test acc = {:.2f}%".format(acc*100))
    


if __name__ == "__main__":
    do_complete_test("dynamic_star_rgb_hand_MCLSTM_9664.pth")

    # model,_ = IterStarRGBHandModel.load_model("dynamic_star_rgb_hand_9728.pth")
    # params = {
    #         "name":"dynamic_star_rgb_hand_LSTM",
    #         "num_classes":20,
    #         "gpu":0,
    #         "epoch":100,
    #         "batch_size":96,
    #         "max_seq":(12,24),
    #         "n_layers":hp.choice("n_layers", [1,2]),
    #         "hidden_dim":hp.choice("hidden_dim",[512,1024]),
    #         "fc_drop":0.15,
    #         "alpha":0.6,
    #         "window":5,
    #         "clip":5.0,
    #         "lr":hp.choice("lr",[[1e-4, 1e-3],[1e-4, 5e-3]]),
    #         "weight_decay":1e-5,
    #         }
    # utils.bayesian_optimization(model,params,iters = 10)






    # model,params = IterStarRGBHandModel.load_model("dynamic_star_rgb_hand_MCLSTM_val_9733.pth", "mc")
    # utils.log( "Hyperparameters = {}".format(utils.__get_params(params)))
    # params["name"] = "dynamic_star_rgb_hand_MCLSTM"
    # params["fc_drop"]=0.2
    # params["batch_size"] = 96
    # params["lr"] = (1e-4,1e-3)

    # # model,params = IterStarRGBHandModel.load_model("dynamic_star_rgb_hand_SPT_LSTM_9350.pth", "mc")
    # # params["max_seq"] = (24,48)
    # # params["batch_size"] = 64
    # # params["fc_drop"]= 0.2
    # # params["lr"] = (1e-4,1e-3)
    # # params["name"] = "dynamic_star_rgb_hand_USPT_LSTM"

    # loader = starRGB_dataset.create_datasets(num_workers=25, batch_size=params["batch_size"], 
    #                                         alpha = params["alpha"], window = params["window"], 
    #                                          max_size =params["max_seq"][1], spotting = model.output_size == 2)
 
    # results = []
    # model.set_dropout(0.05)
    # acc = model.predict(loader["test"],results)
    # with open('results_{}_{}.pkl'.format(params["name"], int(acc*10000)), 'wb') as file:
    #         pickle.dump(results, file, protocol=pickle.HIGHEST_PROTOCOL)
    # dict_save = {"acc":acc,"params":params, "model":model.state_dict()}
    # torch.save(dict_save, "{}_{}.pth".format(params["name"], int(acc*10000) ))
    # utils.log("Test acc = {:.2f}%".format(acc*100))

    # model_tf = model
    







    





















 
