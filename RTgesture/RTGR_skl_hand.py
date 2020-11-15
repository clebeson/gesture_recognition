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
        self.data_type = data_type

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
        self.loss_fn = nn.NLLLoss(weight = class_weights)  if output_size == 2 else nn.NLLLoss()
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

    def spotting_forward(self, hand, skl, mc_samples = None):
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

        out, hidden = self.rnn_spt(x, self.hidden)
        # self.hidden = hidden.data
        self.hidden = ([h.data for h in hidden])

        out = self.bfc_output_spt(out) 
        out = out.contiguous().view(-1, out.size(-1))
        return  F.log_softmax(out,1) 



    def classifier_forward(self, hand, skl, mc_samples = None):
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
        x = self.soft_att_info([hand_emb, skl_emb])
        x = x.view(batch_size, seq_size,-1)
       

        out, hidden = self.rnn(x, self.hidden)
        self.hidden = ([h.data for h in hidden])

        out = self.bfc_output(out) 
        out = out.contiguous().view(-1, out.size(-1))
        return  F.log_softmax(out,1) 


    def forward(self, hand, skl, mc_samples = None):
        if self.output_size == 2:
            return self.spotting_forward(hand, skl, mc_samples)
        return self.classifier_forward(hand, skl, mc_samples)



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
       
        input_size = 21
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
        hiddens = [128]
        learning_rates= [ 1e-4]
        drops = [0.2]
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
        
        data_train = HandSKLDataset(type = "train", max_seq=max_clip_sequence, \
                                              transform=HandTrasformation(output_size = (40,40), data_aug=True), spotting=output_size==2)


        data_test = HandSKLDataset(type = "test", max_seq= max_clip_sequence if output_size==2 else None , \
                                    transform=HandTrasformation(output_size = (40,40), data_aug=False), spotting=output_size==2)
        test_loader = DataLoader(data_test, \
                        batch_size=1, \
                        pin_memory = True,\
                        shuffle=False, \
                        num_workers=args.workers,\
                        drop_last = False)
       

        start = int(clip*10)
        for h,l,d in product(hiddens, learning_rates, drops):
            args.hidden_dim = h
            params.lr = l
            args.dropout = d

            #clip = i*0.1
            params.clip = 5.0# clip
            logging.info("************* Hidden = {}, LR = {}, Dropout {}**************".format(h,l,d))
            folds = 10
            accuracy = np.zeros((folds))
            base_name,_ = os.path.splitext(args.logfile)
            test_results = []
            dict_save = torch.load("saved_models/model_gesture_rt_8040.00.pth")
            hidden_dim = dict_save["hidden_dim"]
            n_layers = dict_save["num_layers"]
            print(hidden_dim,n_layers)
            model = RTGR(args.data_type, input_size, output_size, hidden_dim,n_layers, mode = args.mode, logstd1 = logstd1, logstd2 = logstd2, pi = pi, dropout = args.dropout)
            model = model.to(self.device)
            model.load_state_dict(dict_save["model"],strict= False)
            model = model.parallelize()
            model.train()
            
            
            data_loader = DataLoader(data_train, \
                        batch_size=batch, \
                        pin_memory = True,\
                        shuffle=True, \
                        num_workers=args.workers,\
                        drop_last = False)
            model = self.train(params,model, data_loader, test_loader, max_clip_sequence, self.device, logging)
            del data_loader
            if args.spotting:
                acc = self.predict_frame_wise(model, test_loader, test_results, logging, args.mode)
            else:
                acc = self.predict_gesture_wise(model, test_loader, test_results, logging, args.mode)

               
        
    def predict_gesture_wise(self, model, dataloader, results, log, mode):
        # for d in [0.1,0.15,0.2,0.25,0.3]:
        # print(d)
        log.info("Predicting gestures...")
        num_class = 15
        model = model.module
        model.to(self.device)
        # model.set_dropout(d)
        model.eval()
        criterion = nn.NLLLoss()
        running_loss = 0
        running_loss = 0
        running_corrects = 0
        total = 1e-18 

        mc_samples = 20 #amount of samples for Monte Carlo Estimation

        with torch.no_grad():
            for data in dataloader:
                hand, skl, video_label = data["images"], data["skeletons"], data["labels"]
                loss = 0
                model.reset_hidden()
                label = video_label.repeat(mc_samples,1).reshape(-1).long()
                
                skl = skl.to(self.device)
                hand = hand.to(self.device)
                out = model(hand,skl,mc_samples)
                out = out.cpu()
            
                probs = out.exp().detach().numpy().reshape(mc_samples,hand.shape[1],-1)
                loss = criterion(out, label)
                label = label[0].numpy()
                pred = np.argmax(probs.mean(0), 1)[-1]
                total += 1
                running_loss += loss.item() 

                results.append({"pred":pred, "label":label,  "probs":probs})
                running_corrects += np.sum(pred == label)
            log.info("---> Prediction loss = {}  accuracy = {:.2f}%".format( running_loss/total, running_corrects/float(total)*100) )
        acc = running_corrects/float(total)  
        model_save = {
                "hidden_dim":model.hidden_dim,
                "num_layers":model.n_layers,
                "model":model.state_dict()
            }
        torch.save(model_save, "saved_models/model_{}_{:.2f}.pth".format("gesture_rt", int(acc*10000)))
        pickle.dump(results, open( "predictions/prediction_{}_{:.2f}.pkl".format("gesture_rt",int(acc*10000)), "wb" ), protocol=2)
        return acc

    def predict_frame_wise(self, model, dataloader, results, log, mode):
        
        log.info("Predicting gestures...")
        num_class = 2
        model = model.module
        model.to(self.device)
        # model.set_dropout(d)
        model.eval()
        criterion = nn.NLLLoss()
        running_loss = 0
        running_loss = 0
        running_corrects = 0
        total = 1e-18 

        mc_samples = 20 #amount of samples for Monte Carlo Estimation

        with torch.no_grad():
            # for video, video_skl, video_label in zip(videos,skeletons,labels):
            model.reset_hidden()
            for data in dataloader:
                #print("load took:",time.time()-start)
                hand, skl, video_label = data["images"], data["skeletons"], data["labels"]
                loss = 0
                label = video_label.repeat(mc_samples,1).reshape(-1).long()
                skl = skl.to(self.device)
                hand = hand.to(self.device)
                out = model.spotting_forward(hand,skl,mc_samples)
                out = out.cpu()
                probs = out.exp().detach().numpy().reshape(mc_samples,hand.shape[1],-1)
                
                loss = criterion(out, label)
                label = video_label.numpy().reshape(-1)
                pred = np.argmax(probs.mean(0), 1)
                total += len(pred)
                running_loss += loss.item() 

                results.append({"pred":pred, "label":label,  "probs":probs})
                running_corrects += np.sum(pred == label)
            log.info("---> Prediction loss = {}  accuracy = {:.2f}%".format( running_loss/(total/len(pred)), running_corrects/float(total)*100) )
        # model.set_dropout(model.dropout)
        acc = running_corrects/float(total)  
        model_save = {
                "hidden_dim":model.hidden_dim,
                "num_layers":model.n_layers,
                "model":model.state_dict()
            }
        torch.save(model_save, "saved_models/model_{}_{:.2f}.pth".format("gesture_SPT", int(acc*10000)))
        pickle.dump(results, open( "predictions/prediction_{}_{:.2f}.pkl".format("gesture_SPT",int(acc*10000)), "wb" ), protocol=2)
        model.reset_hidden()
        return acc
    


    def train(self, args, model, dataloader, validation, max_clip_sequence,device, log):
        #put the model in train mode
        model.train()
        log.info("Training...")
        
        #Hyperparameters
        epochs = args.epoch #30
        batch = args.batch_size#32
        sequence = args.seq #10
        clip = args.clip #5
        lr=args.lr#1e-3
        # model.module.freeze_classifier()
        
        
        if  model.module.output_size == 2:
            ft, clf = model.module.get_spt_parameters()
        else:
            ft, clf = model.module.get_parameters()

        optimizers = [optim.Adam(ft, lr=lr), optim.Adam(clf, lr=lr)]
        schedulers = [StepLR(optimizers[0], step_size = 1, gamma = 0.99),
                        StepLR(optimizers[1], step_size = 1, gamma = 0.99)]
        parameters = ft + clf



        model.module.set_dropout(args.dropout)
        model.module.reset_hidden()
        
        steps = len(dataloader.dataset)//batch
        accuracies = []
        # B = steps
        # C = max_clip_sequence//sequence
        lengthscale = 3.0
        N = len(dataloader.dataset)
        weight_decay = lengthscale**2. * (1. - args.dropout) / (2. * N * args.tau)
        

        print("Weight Decay:", weight_decay)
        model.module.reset_hidden()

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
                #print("load took:",time.time()-start)
                data,skl,label = data["images"].to(device), data["skeletons"].to(device), data["labels"].to(device)
                
                size = data.shape[1] // sequence
                #print(data.shape)
                model.module.reset_hidden()
                for s in range(size):
                    model.zero_grad()
                    seq_label = label[:,s*sequence:(s+1)*sequence]
                    seq_hand = data[:,s*sequence:(s+1)*sequence]
                    seq_skl = skl[:,s*sequence:(s+1)*sequence]
                    seq_label = seq_label.contiguous().view(-1)
                    out = model(seq_hand,seq_skl)

                    NLL, KL = model.module.get_loss(out, seq_label,batch)
                    #weight decay
                    l = sum([(p**2).sum() + p.norm(1)  for p in parameters])

                    # Cost function
                    loss = NLL + weight_decay*l
                    #backpropagate error
                    loss.backward()
                    #clips gradients before update weights
                    nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    #update weights
                    for opt in optimizers:opt.step()

                if model.module.output_size == 2:
                    label = label.view(-1)
                    prob, preds = torch.max(out, 1)

                else:
                    out = out.view(-1, sequence,out.size(1))[:,-1,:]
                    label = seq_label.view(-1, sequence)[:,-1]
                    prob, preds = torch.max(out, 1)

            probs.append(prob.cpu().detach().numpy())
            total += out.size(0)
            running_loss += NLL.item() * out.size(0)
                    #running_kl += kl_term.item() * out.size(0)
            running_corrects += torch.sum(preds == label.data).double()
            start = time.time()
            acc = running_corrects/float(total)
            accuracies.append(acc)

            if acc > 0.6: 
                for schd in schedulers:schd.step()
   

            if epoch > 10 and acc < 0.3:
                break
        
            #log information at each 50 epochs
            if epoch%5==0:
                log.info("---> Epoch {:3d}/{:3d}    loss (NLL/KL) = {:4.4f}/{:4.4f}      accuracy = {:2.2f}%".format(epoch + 1,epochs, running_loss/total, 0, running_corrects/float(total)*100) )
                if acc > 0.8:
                    if model.module.output_size == 2:
                        self.predict_frame_wise(model, validation, [], log, "mc")
                    else:
                        self.predict_gesture_wise(model, validation, [], log, "mc")
                    model.module.reset_hidden()
                    model.train()
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
     optmize()


# RTGR_skl_hand.py --logfile gesture_rt.log --tau 100 --n_layers 2 --mode mc --gesture  --trunc_seq 24 --seq 12 --epoch 200 --batch_size 64 --workers 7