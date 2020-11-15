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
import hyperopt
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hand_dataset import HandDataset, HandTrasformation
torch.manual_seed(30)
np.random.seed(30)
random.seed(30) 





def create_datasets(num_workers = 2, batch_size = 32):

    image_datasets = {
        "train":HandDataset(type="train"),
        "test":HandDataset( type= "test")
    }



    dataloaders = {
                   "test":DataLoader(image_datasets["test"], batch_size=batch_size, pin_memory = False,shuffle=False, num_workers=num_workers),
                   "train":DataLoader(image_datasets["train"], batch_size=batch_size, pin_memory = False, shuffle=True, num_workers=num_workers)}
    
    return dataloaders

def inflate_conv(conv2d,
                 time_dim=3,
                 time_padding=0,
                 time_stride=1,
                 time_dilation=1,
                 center=False):
    # To preserve activations, padding should be by continuity and not zero
    # or no padding in time dimension
    kernel_dim = (time_dim, conv2d.kernel_size[0], conv2d.kernel_size[1])
    padding = (time_padding, conv2d.padding[0], conv2d.padding[1])
    stride = (time_stride, conv2d.stride[0], conv2d.stride[0])
    dilation = (time_dilation, conv2d.dilation[0], conv2d.dilation[1])
    conv3d = torch.nn.Conv3d(
        conv2d.in_channels,
        conv2d.out_channels,
        kernel_dim,
        padding=padding,
        dilation=dilation,
        stride=stride)
    # Repeat filter time_dim times along time dimension
    weight_2d = conv2d.weight.data
    if center:
        weight_3d = torch.zeros(*weight_2d.shape)
        weight_3d = weight_3d.unsqueeze(2).repeat(1, 1, time_dim, 1, 1)
        middle_idx = time_dim // 2
        weight_3d[:, :, middle_idx, :, :] = weight_2d
    else:
        weight_3d = weight_2d.unsqueeze(2).repeat(1, 1, time_dim, 1, 1)
        weight_3d = weight_3d / time_dim

    # Assign new params
    conv3d.weight = nn.Parameter(weight_3d)
    conv3d.bias = conv2d.bias
    return conv3d


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

        model = models.resnet18(pretrained=True)
        features = list(model.children())
        
        conv = features[0]
        self.features = nn.Sequential(*features[1:-1])
        self.conv_left = inflate_conv(conv,
                 time_dim=4,
                 time_padding=0,
                 time_stride=1,
                 time_dilation=1,
                 center=False)
        
        self.conv_right = inflate_conv(conv,
                 time_dim=4,
                 time_padding=0,
                 time_stride=1,
                 time_dilation=1,
                 center=False)
        self.loss_fn = nn.NLLLoss(weight = class_weights)  if output_size == 2 else nn.NLLLoss()
        self.output_size = 15 #12
        self.hidden_dim = hidden_dim #256
        self.n_layers = n_layers #2
        self.sharpen = False
        self.lstm = bl.LSTM( input_size = 512,\
                                    hidden_size = self.hidden_dim, num_layers = self.n_layers, \
                                    batch_first=True,**rnn_args) 
                                    
        self.dropout = dropout
        self.fc = bl.Linear(self.hidden_dim, self.output_size, **last_linear_args)
        self.dp_training = True
        self._baysian_layers = []


    def sampling(self,sampling,store):
        for layer in self._baysian_layers:
            layer.sampling = sampling
            layer.store_samples = store

    def forward(self, x, hidden):
  
        batch_size, seq_size, _, C,N,H,W = x.size()
        hl, hr = x[:,:,0].reshape(-1,C,N,W,W), x[:,:,1].reshape(-1,C,N,W,W)
        hands = self.conv_left(hl) + self.conv_right(hr)
        features = self.features(hands.squeeze())

        # time_distributed = []
        # for i in range(seq_size):
            # time_distributed.append(features.squeeze().unsqueeze(1))
        # features = torch.cat(time_distributed,1)
        features = features.view(batch_size, seq_size,-1)
        out, hidden_out = self.lstm(features, hidden)
        out = self.fc(out) 
        out = out.contiguous().view(-1, out.size(-1))
       
        return  F.log_softmax(out,1) , hidden_out

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
        

        
        return NLL

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
        # os.environ["CUDA_LAUNCH_BLOCKING"]="1"

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
       
        input_size = 65
        output_size = 2 if args.spotting else 15
        n_layers =  args.n_layers

        sequence = args.seq
        clip = args.grad_clip
        max_clip_sequence = args.trunc_seq
        logstd1 = -2
        logstd2 = -4
        pi = 0.2
        epoch = args.epoch
        params.hidden_dim = args.hidden_dim
        params.n_layers = args.n_layers
        params.epoch = epoch
        params.spotting = args.spotting
        params.batch_size = args.batch_size
        params.seq = sequence
        params.clip = clip
        params.mode = args.mode
        
        data_loader = DataLoader(HandDataset(type = "train", max_seq=max_clip_sequence, \
                                transform=HandTrasformation(output_size = (50,50), data_aug=True)), \
                                batch_size=params.batch_size, \
                                pin_memory = False,\
                                shuffle=True, \
                                num_workers=args.workers,
                                drop_last = True)

        test_data, test_label = HandDataset(type = "test", transform=HandTrasformation(output_size = (50,50), data_aug=False)).get_data()
        # dataset = HandsDataset(args.seq)
        # dataset.config_data(spotting = False)
        # test_data, test_label =  dataset.get_test()
        # dataset.encode_train(max_clip_sequence)
        # data_loader = DataLoader(dataset, batch_size=batch, shuffle=True, num_workers=args.workers, pin_memory = True)
        hyperparameters = {
            "hidden_dim":hp.choice('hidden_dim',[128, 512,256,1024]), 
            "lr":hp.choice('lr', [1e-3,1e-4,5e-3, 5e-2]),
            "dropout":hp.uniform('dropout', .1,.25),
            "tau":hp.uniform('tau', 1,10),
        }


        def bayesian_optimization(hypers):
    
            args.hidden_dim = hypers["hidden_dim"]
            params.lr =  hypers["lr"]
            params.dropout =  hypers["dropout"]
            params.tau = hypers["tau"]

            #clip = i*0.1
            params.clip = 1# clip
            logging.info("************* Hidden = {}, LR = {:.5f}, Dropout {:.2f}**************".format(hypers["hidden_dim"],hypers["lr"], hypers["dropout"]))

            base_name,_ = os.path.splitext(args.logfile)
            test_results = []
            model = RTGR(args.data_type, input_size, output_size, params.hidden_dim, n_layers, mode = args.mode, logstd1 = logstd1, logstd2 = logstd2, pi = pi, dropout = params.dropout)
            model = model.to(self.device)
            model = model.parallelize()

            model = self.train(params,model, data_loader, max_clip_sequence, self.device, logging)
            
            if args.spotting:
                acc = self.predict_frame_wise(model, test_data,test_results, logging, args.mode)
            else:
                acc = self.predict_gesture_wise(model, test_data,test_label,test_results, logging, args.mode)
            
            if acc > args.thres_train:
                model_save = {
                    "hidden_dim":hypers["hidden_dim"],
                    "num_layers":args.n_layers,
                    "model":model.state_dict()
                }
                torch.save(model_save, "saved_models/model_{}_{:.2f}.pth".format(base_name, acc*100))
                pickle.dump(test_results, open( "predictions/prediction_{}_{:.2f}.pkl".format(base_name,acc*100), "wb" ), protocol=2)
            return {'loss': -acc, 'status': STATUS_OK, "hyperparameters":hypers}

        trials = Trials()
        best = fmin(fn=bayesian_optimization,
                    space=hyperparameters,
                    algo=tpe.suggest,
                    max_evals=100,
                    trials=trials)
        result= {
            "best":best,
            "trials":trials,
        }
        print(best)
        pickle.dump(result, open( "predictions/trials.pkl", "wb" ), protocol=2)

    
    def predict_gesture_wise(self, model, videos, labels, results, log, mode):
        num_class = 15
        log.info("Predicting gestures...")
        model = model.module
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
            for video, label in zip(videos,labels):
                probs = np.zeros((len(video),mc_samples, num_class))
                loss = 0
                hidden = None if mode != "bbb" else [None]*mc_samples
                label =np.array([label[0]])-1
                
                for i, data in enumerate(video):
                    data = np.expand_dims(data,0)
                    if mode == "bbb":
                        data = torch.from_numpy(data).float().unsqueeze(1)
                        data = data.to(self.device)
                        data_label = torch.tensor(label).long()
                        for mc in range(mc_samples):
                            out, hc = model(data,hidden[mc])
                            out = out.cpu()
                            hidden[mc] = ([h.data for h in hc])
                            probs[i,mc] = out.exp().detach().numpy()
                    else:
                        #creating a batch with same data. It improves the performance of MC samples
                        data_label = torch.from_numpy(np.repeat(label,mc_samples,0))
                        #print(data_label.shape, data_label)
                        data = np.repeat(data,mc_samples,0)
                        data = torch.from_numpy(data).float().unsqueeze(1)
                        data = data.to(self.device)
                        out,hidden = model(data,hidden)
                        out = out.cpu()
                        # hidden = ([h.data.mean(1,keepdim=True).repeat(1,mc_samples,1) for h in hidden])

                        hidden = ([h.data for h in hidden])
                    
                        probs[i] = out.exp().detach().numpy()
                
                #label = label.unsqueeze(0)
                loss = criterion(out, data_label)
                pred = np.argmax(probs.mean(1), 1)[-1]
                data_label = data_label.contiguous().view(-1)
                total += 1
                running_loss += loss.item() 
                data_label = data_label.data.numpy()[0]
                # if pred != label:
                #     print(pred,label)
                results.append({"pred":pred, "label":data_label,  "probs":probs})
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
        model.module.set_dropout(args.dropout)

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

            
            start = time.time()
            for data in dataloader:
                # print(epoch)

                #print("load took:",time.time()-start)
                data,label = data["images"].to(device), data["labels"].to(device)
                model.zero_grad()
                
                
                size = data.shape[1] // sequence
                
                for s in range(size):
                    seq_label = label[:,s*sequence:(s+1)*sequence]
                    seq_data = data[:,s*sequence:(s+1)*sequence]
                    seq_label = seq_label.contiguous().view(-1)

                    out,hidden = model(seq_data, hidden)
                    # print(out.shape)
                    # print(seq_label)
                    
                    # out =[]
                    # hidden_h = []
                    # hidden_c = []
                    
                    # for (o,h) in output:
                    #     out.append(o)
                    #     hidden_h.append(h[0].data)
                    #     hidden_c.append(h[1].data)
                    
                    NLL = model.module.get_loss(out, seq_label)
                    
                    # out = torch.cat(out,0)
                    # hidden = [torch.cat(hidden_h,0),torch.cat(hidden_c,0)]

                    hidden = ([h.data for h in hidden])
                    

         
                    l = None
                    if args.mode == "mc":
                        for p in model.parameters():
                            l = (p**2).sum() if l is None else l + (p**2).sum()
                    else:
                        for p in model.parameters():
                            l = p.norm(1) if l is None else l + p.norm(1)
                    
                    loss = NLL + weight_decay*l
                    loss.backward()
                    #clips gradients before update weights
                    # nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                    #update weights
                    optimizer.step()

                hidden = None
                out = out.view(-1, sequence,out.size(1))[:,-1,:]
                label = seq_label.view(-1, sequence)[:,-1]
                prob, preds = torch.max(out, 1)
                probs.append(prob.cpu().detach().numpy())
                total += out.size(0)
                running_loss += loss.item() * out.size(0)
                #running_kl += kl_term.item() * out.size(0)
                running_corrects += torch.sum(preds == label.data).double()
            start = time.time()
            acc = running_corrects/float(total)
            accuracies.append(acc)

            if acc > 0.6: scheduler.step()

            if epoch > 30 and acc < 0.3:
                break
            
            #log information at each 50 epochs
            if epoch%20==0:log.info("---> Epoch {:3d}/{:3d}    loss (NLL/KL) = {:4.4f}/{:4.4f}      accuracy = {:2.2f}%".format(epoch + 1,epochs, running_loss/total, 0, running_corrects/float(total)*100) )
        
        log.info("---> Epoch {:3d}/{:3d}    loss (NLL/KL) = {:4.4f}/{:4.4f}      accuracy = {:2.2f}%".format(epoch + 1,epochs, running_loss/total, 0, running_corrects/float(total)*100) )
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
