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
from dataset import ISDataset
from itertools import product
import torch.optim as optim
import matplotlib.pyplot as plt
import utils
import logging
import threading
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable
import bayesian_layers as bl
import random 
from torch.utils.data import Dataset, DataLoader
from skeleton import *
torch.manual_seed(30)
np.random.seed(30)
random.seed(30) 

class GaussianNoise(nn.Module):
    def __init__(self, stddev):
        super(GaussianNoise,self).__init__()
        self.stddev = stddev

    def forward(self, x):
        return x + torch.randn_like(x) 

class RTGR(nn.Module):
    
    def __init__(self, data_type, input_size, output_size, hidden_dim,n_layers, mode = "mc", dropout = 0.5, logstd1 = -1, logstd2 = -2, pi = 0.5 ):
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
             "type":type,
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
    

    def get_loss(self, output, targets, batch):
        """
        return:
            NLL: NLL is averaged over seq_len and batch_size
            KL: KL is the original scale KL
        """
        # NLL (classification loss)
        NLL = self.get_nll(output, targets)
        

        # KL divergence between posterior and variational prior distribution
        KL =  Variable(torch.zeros(1)).to(output.device) 
        
        for layer in self.get_baysian_layers(): 
            KL += layer.get_kl()
        
        return NLL, KL/batch  #output.size(32)

class optmize:     
    def __init__(self,id=0,file_id=0, args = None):
        #self.random_search(id,file_id)
        self.run_experiment()
       
    def run_experiment(self):
        class param:pass
        args = utils.parse_args()
        params = param()
        logging.basicConfig(
        filename = "{}/{}".format("model_logs",args.logfile),
        level=logging.INFO,
        format='%(asctime)s.%(msecs)03d :: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        )
        os.environ["CUDA_VISIBLE_DEVICES"]= ",".join(str(x) for x in args.devices)

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
       
        input_size = 65
        output_size = 2 if args.spotting else 15
        n_layers =  args.n_layers

        #train params
        lr = args.lr
        batch = args.batch_size
        sequence = args.seq
        clip = args.grad_clip
        max_clip_sequence = args.trunc_seq
        logstd1 = -2
        logstd2 = -4
        pi = 0.2
        hiddens = [512,256,1024]
        learning_rates= [5e-3,1e-3, 1e-2]
        drops = [0.1, 0.12, 0.15, 0.2]
        epoch = args.epoch
        params.hidden_dim = args.hidden_dim
        params.n_layers = args.n_layers
        params.tau = args.tau
        params.dropout = args.dropout
        params.epoch = epoch
        params.spotting = args.spotting
        params.lr = lr
        params.batch_size = batch
        params.seq = sequence
        params.clip = clip
        params.mode = args.mode
        
        dataset = ISDataset(args.seq)
        dataset.config_data(spotting = False)
        test_data, test_label =  dataset.get_test()
        data_loader = DataLoader(dataset, batch_size=batch, shuffle=True, num_workers=args.workers)
        

        start = int(clip*10)
        for h,l,d in product(hiddens, learning_rates, drops):
            args.hidden_dim = h
            params.lr = l
            args.dropout = d

            #clip = i*0.1
            params.clip = 1# clip
            logging.info("************* Hidden = {}, LR = {}, Dropout {}**************".format(h,l,d))
            folds = 10
            accuracy = np.zeros((folds))
            base_name,_ = os.path.splitext(args.logfile)
            test_results = []
            # for k, (train_data, test_data) in  enumerate(dataset.cross_validation(k=folds, spotting = args.spotting)): 
            #     model = RTGR(args.data_type, input_size, output_size, args.hidden_dim,n_layers, mode = args.mode, logstd1 = logstd1, logstd2 = logstd2, pi = pi, dropout = args.dropout)
            #     model = model.to(self.device)
            #     model = self.train(params,model, train_data, max_clip_sequence, self.device, logging)
            #     if args.spotting:
            #         acc = self.predict_frame_wise(model, test_data,test_results, logging, args.mode)
            #     else:
            #         acc = self.predict_gesture_wise(model, test_data,test_results, logging, args.mode)

            #     accuracy[k] = acc
            #     if acc < args.thres_train:break
                
            #     if acc > 0.9:
            #         model_save = {
            #             "hidden_dim":h,
            #             "num_layers":args.n_layers,
            #             "model":model.state_dict()
            #         }
            #         torch.save(model_save, "saved_models/model_{}_{}_{:.2f}.pth".format(base_name,k,acc*100))
            #         pickle.dump(test_results, open( "predictions/prediction_{}_{}_{:.2f}.pkl".format(base_name,k,acc*100), "wb" ), protocol=2)
            model = RTGR(args.data_type, input_size, output_size, args.hidden_dim,n_layers, mode = args.mode, logstd1 = logstd1, logstd2 = logstd2, pi = pi, dropout = args.dropout)
            model = model.to(self.device)
            
            model = self.train(params,model, data_loader, max_clip_sequence, self.device, logging)
            
            if args.spotting:
                acc = self.predict_frame_wise(model, test_data,test_results, logging, args.mode)
            else:
                acc = self.predict_gesture_wise(model, test_data,test_label,test_results, logging, args.mode)
            
            if acc > args.thres_train:
                model_save = {
                    "hidden_dim":h,
                    "num_layers":args.n_layers,
                    "model":model.state_dict()
                }
                torch.save(model_save, "saved_models/model_{}_{:.2f}.pth".format(base_name, acc*100))
                pickle.dump(test_results, open( "predictions/prediction_{}_{:.2f}.pkl".format(base_name,acc*100), "wb" ), protocol=2)


    
    def predict_gesture_wise(self, model, videos, labels, results, log, mode):
        num_class = 15
        log.info("Predicting gestures...")
        model.to(self.device)
        model.set_dropout(0.1)
        model.eval()
        criterion = nn.NLLLoss()
        running_loss = 0
        running_loss = 0
        running_corrects = 0
        total = 1e-18 

        mc_samples = 20 #amount of samples for Monte Carlo Estimation

        with torch.no_grad():
            for (video, interval), label in zip(videos,labels):
                probs = np.zeros((len(video),mc_samples, num_class))
                loss = 0
                hidden = None if mode != "bbb" else [None]*mc_samples
                label = np.expand_dims(label[0],0)

                for i, data in enumerate(video):
                    data = np.expand_dims(data,0)
                    if mode == "bbb":
                        data = torch.from_numpy(data).float().unsqueeze(1)
                        data = data.to(self.device)
                        label = torch.tensor(label).long()
                        for mc in range(mc_samples):
                            out, hc = model(data,hidden[mc])
                            out = out.cpu()
                            hidden[mc] = ([h.data for h in hc])
                            probs[i,mc] = F.log_softmax(out,1).exp().detach().numpy()
                    else:
                        #creating a batch with same data. It improves the performance of MC samples
                        label = np.repeat(label,mc_samples,0)
                        data = np.repeat(data,mc_samples,0)
                        label = torch.tensor(label).long()
                        data = torch.from_numpy(data).float().unsqueeze(1)
                        data = data.to(self.device)
                        out,hidden = model(data,hidden)
                        out = out.cpu()
                        # hidden = ([h.data.mean(1,keepdim=True).repeat(1,mc_samples,1) for h in hidden])

                        hidden = ([h.data for h in hidden])
                    
                        probs[i] = F.log_softmax(out,1).exp().detach().numpy()

                #label = label.unsqueeze(0)
                loss = criterion(F.log_softmax(out,1), label)
                pred = np.argmax(probs.mean(1), 1)[-1]
                label = label.contiguous().view(-1)
                total += 1
                running_loss += loss.item() 
                label = label.data.numpy()[0]
                # if pred != label:
                #     print(pred,label)
                results.append({"pred":pred, "label":label,  "probs":probs, "interval":interval})
                running_corrects += np.sum(pred == label)

            log.info("---> Prediction loss = {}  accuracy = {:.2f}%".format( running_loss/total, running_corrects/float(total)*100) )
        return running_corrects/float(total)  

    def predict_frame_wise(self, model, test_data, results, log, mode):
        num_class = 2
        log.info("Predicting Frame Wise...")
        model.to(self.device)
        model.eval()
        model.sampling(True,False)
        #model.set_dropout(0.15)
        criterion = nn.NLLLoss()
        running_loss = 0
        running_loss = 0
        running_corrects = 0
        total = 1e-18 

        mc_samples = 100 #amount of samples for Monti Carlo Estimation

        with torch.no_grad():
                probs = np.zeros((len(test_data),mc_samples, num_class))
                loss = 0
                hidden = None
                labels = []
                for i, data in enumerate(test_data):
                    label = data[0].astype(int)
                    label = 0 if label<=0 else 1
                    
                    labels.append(label)
                    label = np.expand_dims(label,0)
                    data = data[1:]
                    data = np.expand_dims(data,0)
                    
                    #creating a batch with same data. It improves the performance of MC samples
                    label = np.repeat(label,mc_samples,0)
                    data = np.repeat(data,mc_samples,0)
                    label = torch.tensor(label).long()
                    data = torch.from_numpy(data).float().unsqueeze(1)
                    data = data.to(self.device)
                    out,hidden = model(data,hidden)
                    out = out.cpu()
                    # hidden = ([h.data for h in hidden])
                    hidden = ([h.data.mean(1,keepdim=True).repeat(1,mc_samples,1) for h in hidden])

                    
                    
                    loss = criterion(F.log_softmax(out,1), label)
                    probs[i] = F.log_softmax(out,1).exp().detach().numpy()
                    running_loss += loss.item() 

                #label = label.unsqueeze(0)
                labels = np.array(labels)
                pred = np.argmax(probs.mean(1), 1)
                label = label.contiguous().view(-1)
                total = len(labels)
                label = label.data.numpy()[0]
                # if pred != label:
                #     print(pred,label)
                results.append({"pred":pred, "label":labels,  "probs":probs})
                running_corrects += np.sum(pred == labels)
        
        log.info("---> Prediction loss = {}  accuracy = {:.2f}%".format( running_loss/total, running_corrects/float(total)*100) )
        return running_corrects/float(total)    
    


    def train(self, args, model, dataloader, max_clip_sequence,device, log):
        #put the model in train mode
        model.train()
        log.info("Training...")
        
        #Hyperparameters
        epochs = args.epoch #30
        batch = args.batch_size#32
        sequence = args.seq #10
        clip = args.clip #5
        lr=args.lr#1e-3
        model.set_dropout(args.dropout)

        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = StepLR(optimizer, step_size=1, gamma=0.99)
        steps = len(dataloader.dataset)//batch
        accuracies = []
        # B = steps
        # C = max_clip_sequence//sequence
        lengthscale = 3.0
        N = len(dataloader.dataset)
        weight_decay = lengthscale**2. * (1. - args.dropout) / (2. * N * args.tau)
        

        print("Weight Decay:", weight_decay)

        for epoch in range(epochs):
            running_loss = 0
            running_kl = 0
            running_corrects = 0 
            total = 1e-18
            #importance of KL-divergence terms in relation to classification loss
            scale = 1.0
            hidden = None
            probs = []

            
            
            for data in dataloader:
                data,label = torch.from_numpy(data["skeletons"]).to(device), torch.from_numpy(data["labels"]).to(device)
                model.zero_grad()
                label = label.contiguous().view(-1)-1
                

                size = data.shape[1] // sequence
                
                for s in range(size):
                    seq_label = label[:,s*sequence:(s+1)*sequence]
                    seq_data = data[:,s*sequence:(s+1)*sequence]
                    outputs = []

                    out, hidden = model(data, hidden)
                    hidden = ([h.data for h in hidden])
                    

                    # proper scaling for sequence loss
                    NLL, KL = model.get_loss(out, label,batch)
                    NLL_term = NLL
                    l = None
                    if args.mode == "mc":
                        for p in model.parameters():
                            l = (p**2).sum() if l is None else l + (p**2).sum()
                    else:
                        for p in model.parameters():
                            l = p.norm(1) if l is None else l + p.norm(1)
                    
                    NLL_term = NLL_term + weight_decay*l
                    # proper scaling for sequence loss
                    #Composed loss
                    loss = NLL_term
                    #backpropagate error
                    loss.backward()
                    #clips gradients before update weights
                    # nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                    #update weights
                    optimizer.step()

                hidden = None
                out = out.view(-1, sequence,out.size(1))[:,-1,:]
                label = label.view(-1, sequence)[:,-1]
                prob, preds = torch.max(out, 1)
                probs.append(prob.cpu().detach().numpy())
                total += out.size(0)
                running_loss += NLL_term.item() * out.size(0)
                running_kl += kl_term.item() * out.size(0)
                running_corrects += torch.sum(preds == label.data).double()
            
            
            probs = np.array(probs)
            time = np.arange(len(probs))
            
            acc = running_corrects/float(total)
            accuracies.append(acc)

            if acc > 0.6: scheduler.step()

            if epoch > 99 and acc < 0.3:
                break

            #log information at each 50 epochs
            if epoch%100==0:log.info("---> Epoch {:3d}/{:3d}    loss (NLL/KL) = {:4.4f}/{:4.4f}      accuracy = {:2.2f}%".format(epoch + 1,epochs, running_loss/total, running_kl/total, running_corrects/float(total)*100) )
        
        log.info("---> Epoch {:3d}/{:3d}    loss (NLL/KL) = {:4.4f}/{:4.4f}      accuracy = {:2.2f}%".format(epoch + 1,epochs, running_loss/total, running_kl/total, running_corrects/float(total)*100) )
        #return a trained model
        return model



 #hyperparameters tuning       
def hyper_tuning():
    def opt(id=0,file_id=0):
        optmize(id,file_id)
    def has_live_threads(threads):
        return True in [t.isAlive() for t in threads]
    threads = []
    #distribuite the random search of hyperparameters over 3 threads. In this case, one per available GPU
    for index in [[0,0],[1,1],[2,2]]:
        x = threading.Thread(target=opt, args=tuple(index))
        threads.append(x)
        x.start()

    while has_live_threads(threads):
        try:
            # synchronization timeout of threads kill
            [t.join(1) for t in threads
             if t is not None and t.isAlive()]
        except KeyboardInterrupt:
            # Ctrl-C handling and send kill to threads
            print("Sending kill to threads...")
            for t in threads:
                t.kill = True
            sys.exit(1) 
    

            
if __name__ == "__main__":
     #hyper_tuning()
     optmize()

