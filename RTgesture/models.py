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
from acticipate import ActicipateDataset
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
warnings.filterwarnings(action='ignore', category=ConvergenceWarning)
torch.manual_seed(30)
np.random.seed(30)
random.seed(30) 


class RTGR(nn.Module):
    
    def __init__(self, data_type, input_size, output_size, hidden_dim,n_layers, mode = "mc", dropout = 0.5, logstd1 = -1, logstd2 = -2, pi = 0.5 ):
        super(RTGR, self).__init__()
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
        self.data_type = data_type

        #self.norm = nn.InstanceNorm1d(input_size)
        #Embedding

        self.input_size = input_size
        self.embeding_size = 16
        self.bfc1 = bl.Linear(self.input_size, 32, **linear_args)
        self.bfc2 = bl.Linear(32, self.embeding_size, **linear_args)
        self.fc1 = nn.Sequential(self.bfc1,nn.ReLU())
        self.fc2 = nn.Sequential(self.bfc2,nn.ReLU())
        #weights = [1./21, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,1.0, 1.0, 1.0, 1.0, 1.0]
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


    def sampling(self,sampling,store):
        for layer in self._baysian_layers:
            layer.sampling = sampling
            layer.store_samples = store

    def forward(self, x, hidden):
        # print(x[0,0].sum())
        # x = self.noise(x)
        # x = self.norm1(x.view(-1,self.input_size)).view(x.shape)
        x = self.fc1(x)
        emb = self.fc2(x)
        # emb = self.norm2(emb.view(-1,self.embeding_size)).view(emb.shape)
        out, hidden_out = self.lstm(emb, hidden)
        out = self.fc(out) 
   
        # if targets is not None:
        #     out = out.contiguous().view(-1, out.size(-1))
        #     out, hidden_out = self.sharpening_posterior(emb,hidden,out,targets)
        #     out = self.fc(out) 
        
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
        # NLL (classification loss)
        NLL = self.get_nll(output, targets)
        return NLL, None

        # # KL divergence between posterior and variational prior distribution
        # KL =  Variable(torch.zeros(1)).to(output.device) 
        
        # for layer in self.get_baysian_layers(): 
        #     KL += layer.get_kl()
        
        # return NLL, KL/batch  #output.size(32)
        
class KnnDtw(object):
    def __init__(self, n_neighbors=10, max_warping_window=10, subsample_step=1):
        self.n_neighbors = n_neighbors
        self.max_warping_window = max_warping_window
        self.subsample_step = subsample_step
    
    def fit(self, x, l):
        self.x = x
        self.l = l
        return self
        
    def _dtw_distance(self, ts_a, ts_b, d = lambda x,y: np.dot(x,y)):
        # Create cost matrix via broadcasting with large int
        ts_a, ts_b = np.array(ts_a), np.array(ts_b)
        M, N = len(ts_a), len(ts_b)
        cost = float("inf") * np.ones((M, N))

        # Initialize the first row and column
        cost[0, 0] = d(ts_a[0], ts_b[0])
        for i in range(1, M):
            cost[i, 0] = cost[i-1, 0] + d(ts_a[i], ts_b[0])

        for j in range(1, N):
            cost[0, j] = cost[0, j-1] + d(ts_a[0], ts_b[j])

        # Populate rest of cost matrix within window
        for i in range(1, M):
            for j in range(max(1, i - self.max_warping_window),
                            min(N, i + self.max_warping_window)):
                choices = cost[i - 1, j - 1], cost[i, j-1], cost[i-1, j]
                cost[i, j] = min(choices) + d(ts_a[i], ts_b[j])

        # Return DTW distance given window 
        # print(cost[-1, -1])
        return cost[-1, -1]
    
    def _dist_matrix(self, x, y):
        # Compute the distance matrix        
        dm_count = 0
        
        # Compute condensed distance matrix (upper triangle) of pairwise dtw distances
        # when x and y are the same array
        if(np.array_equal(x, y)):
            x_s = np.shape(x)
            dm = np.zeros((x_s[0] * (x_s[0] - 1)) // 2, dtype=np.double)
            
            # p = ProgressBar(shape(dm)[0])
            
            for i in range(0, x_s[0] - 1):
                for j in range(i + 1, x_s[0]):
                    dm[dm_count] = self._dtw_distance(x[i, ::self.subsample_step],
                                                      y[j, ::self.subsample_step])
                    
                    dm_count += 1
                    # p.animate(dm_count)
            
            # Convert to squareform
            dm = squareform(dm)
            return dm
        
        # Compute full distance matrix of dtw distnces between x and y
        else:
            x_s = np.shape(x)
            y_s = np.shape(y)
            dm = np.zeros((x_s[0], y_s[0])) 
            dm_size = x_s[0]*y_s[0]
            
            # p = ProgressBar(dm_size)
        
            for i in range(0, x_s[0]):
                for j in range(0, y_s[0]):
                    dm[i, j] = self._dtw_distance(x[i, ::self.subsample_step],
                                                  y[j, ::self.subsample_step])
                    # Update progress bar
                    dm_count += 1
                    # p.animate(dm_count)
        
            return dm
        
    def predict(self, x):
      
        dm = self._dist_matrix(x, self.x)

        # Identify the k nearest neighbors
        knn_idx = dm.argsort()[:, :self.n_neighbors]

        # Identify k nearest labels
        knn_labels = self.l[knn_idx]
        
        # Model Label
        mode_data = mode(knn_labels, axis=1)
        mode_label = mode_data[0]
        mode_proba = mode_data[1]/self.n_neighbors

        return mode_label.ravel(), mode_proba.ravel()



class RNNModel(nn.Module):
    def __init__(self,params):
        super(RNNModel, self).__init__()

        self.rnn = nn.LSTMCell(26,params["rnn_hidden"], bias=True)
        self.clf = nn.Sequential(nn.Linear(int(params["rnn_hidden"]), params["mlp_hidden"], bias = True),
                 nn.ReLU(), nn.Linear( params["mlp_hidden"], params["num_classes"]), nn.LogSoftmax(1))
        self.hx = None
        
        
    def forward(self,x):
        seq = x.size(1)
        outs = []
        for s in range(seq):
            hx, cx = self.rnn(x[:,s,:], self.hx)
            self.hx = (hx.data,cx.data)
            outs.append(hx.unsqueeze(1))
        outs = self.clf(torch.cat(outs,1))
        return outs
        

class ConvModel(nn.Module):
    def __init__(self,params):
        super(ConvModel, self).__init__()
        max = params["max_seq"]
        self.conv = nn.Sequential(
            nn.Conv1d(params["num_features"], params["out_ch1"], params["kernel1"], stride=1, bias = True),
            nn.ReLU(), nn.MaxPool1d(2,2),
            nn.Conv1d(params["out_ch1"], params["out_ch2"], params["kernel2"], stride=1,bias = True),
            nn.ReLU(), nn.MaxPool1d(2,2),
            nn.Conv1d(params["out_ch2"], params["out_ch3"], params["kernel3"], stride=1, bias = True),
            nn.ReLU(), nn.MaxPool1d(2,2)
        )

        for k in [params["kernel1"],params["kernel2"],params["kernel3"]]:
            n = (max - k)/2.
            max = n if int(n) == n else int(n)+1
        # print(max)
        self.clf = nn.Sequential(nn.Linear(int(params["out_ch3"]*max), params["hidden_layer"], bias = True),
                 nn.ReLU(), nn.Linear( params["hidden_layer"], params["num_classes"]), nn.LogSoftmax(1))
        
        
    def forward(self,x):
        # print(x.shape)
        x = self.conv(x)
        x = x.view(x.size(0),-1)
        return self.clf(x)


class GaussianNoise(nn.Module):
    def __init__(self, stddev):
        super(GaussianNoise,self).__init__()
        self.stddev = stddev

    def forward(self, x):
        return x + torch.randn_like(x) 


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


def predict_DLSTM(model, videos, params, results):
    device = torch.device('cuda:{}'.format(params["gpu"]) if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    running_corrects = 0
    with torch.no_grad():
        for (video, label, (b,e)) in videos:
            data = torch.from_numpy(video).float().unsqueeze(0)
            data = data.to(device)
            out, _ = model(data,None)
            out = out.cpu()
            probs = F.log_softmax(out,1).exp().detach().numpy()
            pred = np.argmax(probs, 1)[-1]
            running_corrects += np.sum(pred == label)
            results.append({"pred":pred, "label":label,  "probs":probs, "interval":(b,e)})
    return running_corrects/len(videos)    

def predict_BBBLSTM(model, videos, params, results):
    device = torch.device('cuda:{}'.format(params["gpu"]) if torch.cuda.is_available() else 'cpu')
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

def train_LSTM( train_data, params):
    device = torch.device('cuda:{}'.format(params["gpu"]) if torch.cuda.is_available() else 'cpu')
    model = RTGR(data_type =  params["data_type"], 
                input_size =  params["input_size"], 
                output_size =  params["num_classes"], 
                hidden_dim =  params["hidden_dim"], 
                n_layers =  params["n_layers"],  
                mode =  "DET",
                dropout = params["fc_drop"], 
                )

    model = model.to(device)
    model.train()
    # model.set_dropout(0.0)
    #Hyperparameters
    epochs = params["epoch"]
    batch = int(len(train_data)*(1.0/params["batch_size"]))
    sequence = params["max_seq"][0] 
    clip = params["clip"]
    lr=params["lr"]
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.99)
    indexes = list(range(len(train_data)))
    accuracies = []
    accuracies = []
    B = len(train_data)//batch
    C = params["max_seq"][1] //sequence
    loss = 0
    for epoch in range(epochs):
        entered = False
        random.shuffle(indexes)
        train = [ train_data[i] for i in indexes]
        running_loss = 0
        running_kl = 0
        running_corrects = 0 
        total = 1e-18
        #importance of KL-divergence terms in relation to classification loss
        scale = 0.1
        hidden = None
        probs = []
        for data, label,end_sequence_batch in to_batch(train,seq=sequence, batch_size = batch, max = params["max_seq"][1]):
            entered = True
            data,label = torch.from_numpy(data).float().to(device), torch.from_numpy(label).to(device)
            model.zero_grad()
            label = label.contiguous().view(-1)
            outputs = []
            out, hidden = model(data, hidden)
            hidden = ([h.data for h in hidden])

            if model.type == bl.ModelType.MC_DROP or  model.type == bl.ModelType.DET:
                l2 = None
                for p in model.parameters():
                    l2 = (p**2).sum() if l2 is None else l2 + (p**2).sum()
            
                NLL, _ = model.get_loss(out, label)
                # proper scaling for sequence loss
                loss = NLL   + params["weight_decay"] * l2
                #backpropagate error
            else:
                NLL, KL = model.get_loss(out, label)
                # proper scaling for sequence loss
                NLL_term = NLL / C
                # proper scaling for sequence loss
                kl_term = scale*(KL / B*C)
                #Composed loss
                loss = NLL_term + kl_term 
            
            loss.backward()
            #clips gradients before update weights
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            #update weights
            optimizer.step()


            if end_sequence_batch:
                hidden = None
                out = out.view(-1, sequence,out.size(1))[:,-1,:]
                label = label.view(-1, sequence)[:,-1]
                prob, preds = torch.max(out, 1)
                probs.append(prob.cpu().detach().numpy())
                total += out.size(0)
                running_loss += loss.item() * out.size(0)
                running_kl += loss.item() * out.size(0)
                running_corrects += torch.sum(preds == label.data).double()
        
        
        probs = np.array(probs)
        time = np.arange(len(probs))
        acc = running_corrects/float(total)
        accuracies.append(acc)
        
        if epoch >150 and acc < 50: break
        if acc > 0.2: scheduler.step()
        
        # if entered:
        #     print("Epoch = {}  Train acc = {:.2f}   loss = {:.2f}".format(epoch,acc.cpu().numpy(), loss.item()))

        if acc == 1.0: break
        
        #early stop condition (mean accuracy over the last 10 training epochs) > 99%
        if len(accuracies)  > 5: 
            del accuracies[0]
            mean = sum(accuracies)/float(len(accuracies))
            if  mean > 0.99: break
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




            
# if __name__ == "__main__":
#     #hyper_tuning()
#     optmize()

#100 python action_anticipation.py --seq 120 --batch_size 216 --hidden_dim  64 --lr 1e-2 --epoch 200 --n_layers 1 --trunc_seq 120 --grad_clip 4.7
#99.58 python action_anticipation.py --seq 120 --batch_size 200 --hidden_dim  64 --lr 1e-2 --epoch 200 --n_layers 2 --trunc_seq 120 --grad_clip .6
#98.75 python action_anticipation.py --seq 32 --batch_size 32 --hidden_dim  256 --lr 1e-3 --epoch 200 --n_layers 4 --trunc_seq 128 --grad_clip 5
#ball 95.42 python action_anticipation.py --seq 120 --batch_size 216 --hidden_dim  64 --lr 1e-2 --epoch 200 --n_layers 2 --trunc_seq 120 --grad_clip .5
# remain python action_anticipation.py --seq 120 --batch_size 216 --hidden_dim  64 --lr 1e-2 --epoch 200 --n_layers 2 --trunc_seq 120 --grad_clip 5
# 6 class python action_anticipation.py --seq 120 --batch_size 108 --hidden_dim  32 --lr 2e-2 --epoch 200 --n_layers 1 --trunc_seq 120 --data_type g --grad_clip 0.15

#python action_anticipation.py --seq 100 --batch_size 216 --hidden_dim 64 --lr 1e-1 --epoch 2000 --n_layers 2 --trunc_seq 100 --grad_clip 0.2

#input_size = 32, output_size = 12, hidden_dim = 32, n_layers = 2, std = -4, epoch= 1000,lr = 0.05,batch = 16,sequence = 8,clip = 0.7, max=100