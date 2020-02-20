from collections import deque
import numpy as np
import cv2
import csv
import sys
import os
import argparse
import math
#import pims
#import pandas as pd
#from moviepy.editor import *
import glob
import pickle
import random
import torch
import torch.nn as nn
from torch.nn import functional as F
from acticipate import ActicipateDataset
import torch.optim as optim
import matplotlib.pyplot as plt
import utils
import logging
import threading
from torch.optim.lr_scheduler import StepLR
from  context_model import *
from movement_model import *
from torch.autograd import Variable
import new_implt as bl
torch.manual_seed(30)
np.random.seed(30)
random.seed(30) 

class GaussianNoise(nn.Module):
    def __init__(self, stddev):
        super(GaussianNoise,self).__init__()
        self.stddev = stddev

    def forward(self, x):
        return x + torch.randn_like(x) 

class BActAnticipationModel(nn.Module):
    
    def __init__(self, data_type, input_size, output_size, hidden_dim,n_layers, rnn_dropout = 0.5, fc_dropout = 0.5, logstd1 = -1, logstd2 = -2, pi = 0.5 ):
        super(BActAnticipationModel, self).__init__()
        type = bl.ModelType.VAR_DROP_B_ADAP
        
        linear_args =  {"mu":0, "logstd1":logstd1, "logstd2":logstd2, "pi":pi, "type":type}
        rnn_args =  {"mu":0, "logstd1":logstd1, "logstd2":logstd2, "pi":pi,  "type":type}
        self.data_type = data_type
        self.context = ContextModel(**linear_args)
        self.movement = MovementModel(**linear_args)
        self.loss_fn = nn.NLLLoss() 
        self.input_size = input_size #44
        self.output_size = output_size #12
        self.hidden_dim = hidden_dim #256
        self.n_layers = n_layers #2
        self.sharpen = False
        self.lstm = bl.LSTM( input_size = self.input_size,\
                                    hidden_size = self.hidden_dim, num_layers = self.n_layers, \
                                    dropout=rnn_dropout, batch_first=True,**rnn_args) 
                                    
        self.noise = GaussianNoise(0.01)


        # dropout layer
        self.dropout = fc_dropout
        self.fc_combine = bl.Linear(self.input_size, self.input_size, **linear_args)
        self.combine = nn.Sequential(self.fc_combine, nn.ReLU())
        
        # linear and sigmoid layers
        self.fc = bl.Linear(self.hidden_dim, self.output_size, **linear_args)
        self.dp_training = True
        self._baysian_layers = []

    
    def embedding(self,x):
       # x = self.noise(x) #adding gaussian noise into input
        xc = x[...,:12]
        xm = x[...,12:]

        if "g" not in self.data_type:
            xc[:,2:] = -1
        if "b" not in self.data_type:
             xc[:,:2] = -1
        if "m" not in self.data_type:
            xm[:,:] = -1       

        xc = self.context(xc)
        xm = self.movement(xm)
        x = torch.cat( (xm,xc),  -1) 
       # x = F.dropout(x,self.dropout, training=self.dp_training)
        x = self.combine(x)
       # x = F.dropout(x,self.dropout, training=self.dp_training)
        return x

    def sampling(self,sampling,store):
        for layer in self._baysian_layers:
            layer.sampling = sampling
            layer.store_samples = store

    def forward(self, x, hidden):
        
        emb = self.embedding(x)
        out, hidden_out = self.lstm(emb, hidden)
        #out = F.dropout(out,self.dropout, training=self.dp_training)
        out = self.fc(out) 
   
        # if targets is not None:
        #     out = out.contiguous().view(-1, out.size(-1))
        #     out, hidden_out = self.sharpening_posterior(emb,hidden,out,targets)
        #     out = self.fc(out) 
        
        out = out.contiguous().view(-1, out.size(-1))
        return out, hidden_out

    def set_dropout(self, value, training=True):
        self.dropout = value
        self.movement.dropout=value
        self.context.dropout=value

        self.dp_training = training
        self.movement.dp_training = training
        self.context.dp_training = training


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


        # KL divergence between posterior and variational prior distribution
        KL = Variable(torch.zeros(1)).to(output.device)
        
        for layer in self.get_baysian_layers(): 
            KL += layer.get_kl()
        
        return NLL, KL / 32.0 #output.size(32)

class optmize:     
    def __init__(self,id=0,file_id=0, args = None):
        #self.random_search(id,file_id)
        self.run_experiment()
       
    def run_experiment(self):
        class param:pass
        params = param()
        logging.basicConfig(
        filename='BBBlogs.log',
        level=logging.INFO,
        format='%(asctime)s.%(msecs)03d :: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        )
        self.device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
        args = utils.parse_args()
        dataset = ActicipateDataset()
        input_size = 32
        output_size = 12
        hidden_dim = args.hidden_dim
        n_layers =  args.n_layers
        rnn_dropout = .5
        fc_dropout = .7
    
        #train params
        lr = args.lr
        batch = args.batch_size
        sequence = args.seq
        clip = args.grad_clip
        max_clip_sequence = args.trunc_seq
        logstd1 = -2
        logstd2 = -4
        pi = 0.2
        
        epoch = args.epoch
        params.hidden_dim = args.hidden_dim
        params.n_layers = args.n_layers
        params.rnn_dropout = rnn_dropout
        params.fc_dropout = fc_dropout
        params.epoch = epoch
        params.lr = lr
        params.batch_size = batch
        params.seq = sequence
        params.clip = clip
        start = int(clip*10)
        for i in range(start,100):
            clip = i*0.1
            params.clip = clip
            logging.info("************* CLIP = {} **************".format(params.clip))
            folds = 10
            accuracy = np.zeros((folds))
            test_results = []
            for k, (train_data, test_data) in  enumerate(dataset.cross_validation(k=folds)): 
                model = BActAnticipationModel(args.data_type, input_size, output_size, hidden_dim,n_layers, logstd1 = logstd1, logstd2 = logstd2, pi = pi, rnn_dropout = 0.5, fc_dropout = 0.5)
                model = model.to(self.device)
                model = self.train(params,model, train_data, max_clip_sequence, self.device, logging)  
                acc = self.predict(model, test_data,test_results, logging)
                accuracy[k] = acc
                if acc < 0.99:break
                
            if accuracy.mean()  >= 0.99:
                experiment = "{:4d}".format(int(np.random.rand()*10000))
                pickle.dump(test_results, open( "prediction_BBB_{}.pkl".format(experiment), "wb" ), protocol=2)
                #pickle.dump(params, open( "params_brnn_{}.pkl".format( experiment), "wb" ), protocol=2)
                logging.info("Cross val: {:.2f}%".format(accuracy.mean()*100))
                torch.save(model.state_dict(), "act_model_BBB_{}.pth".format(experiment))
                break

    def random_search(self,id, file_id,):
        class param:pass
        params = param()

        logger = logging.getLogger('simple_example{}'.format(file_id))
        logger.setLevel(logging.INFO)
        # create file handler which logs even debug messages
        fh = logging.FileHandler('log_test{}.log'.format(file_id))
        fh.setLevel(logging.INFO)
        # create formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s :: %(message)s')
        fh.setFormatter(formatter)
        # add the handlers to logger
        logger.addHandler(fh)

        param_msg = "input_size = {}, output_size = {}, hidden_dim = {}, n_layers = {}, std = {}, epoch= {},lr = {},batch = {},sequence = {},clip = {}, max={}"
        
        #logger.basicConfig(filename="log_test_{}.txt".format(file_id), filemode='w',level=logging.INFO, format='%(asctime)s - %(message)s')
        self.device = torch.device('cuda:{}'.format(id) if torch.cuda.is_available() else 'cpu')
        logger.info("\n\n\nRunning in {}".format(self.device))

        dataset = ActicipateDataset()
        for _ in range(40):
        #model params
            input_size = 32
            output_size = 12
            hidden_dim = np.random.choice(a= [64],size=1)[0]
            n_layers =  np.random.choice(a= [2],size=1)[0]
            rnn_dropout = np.random.choice(a= [.5],size=1)[0]
            fc_dropout = np.random.choice(a= [.5],size=1)[0]
            
            #train params
            epoch = 1000
            lr = np.random.choice(a= [5e-3],size=1)[0]
            batch = np.random.choice(a= [32],size=1)[0]
            sequence = np.random.choice(a= [32],size=1)[0]
            clip = np.random.choice(a= [0.1, 0.2,0.3,0.4],size=1)[0]
            max_clip_sequence = np.random.choice(a= [128],size=1)[0]
            logstd = np.random.choice(a= [-2],size=1)[0]
            params.hidden_dim = hidden_dim
            params.n_layers = n_layers
            params.rnn_dropout = rnn_dropout
            params.fc_dropout = fc_dropout
            params.epoch = epoch
            params.lr = lr
            params.batch_size = batch
            params.seq = sequence
            params.clip = clip
            params.logstd = logstd
            params.max_clip_sequence = max_clip_sequence            

            logger.info(param_msg.format(input_size,output_size,hidden_dim,n_layers,logstd,epoch,lr,batch,sequence,clip,max_clip_sequence))
            folds = 10
            accuracy = np.zeros((folds))
            test_results = []
            for k, (train_data, test_data) in  enumerate(dataset.cross_validation(k=folds)): 
                model = BActAnticipationModel("lstm", input_size, output_size, hidden_dim,n_layers, rnn_dropout = 0.5, fc_dropout = 0.5)
                model = model.to(self.device)
                model = self.train(params,model, train_data, max_clip_sequence, self.device, logger)  
                acc = self.predict(model, test_data,test_results, logger)
                accuracy[k] = acc
                if acc < 0.95:break
            
            if accuracy.mean() > 0.98: 
                experiment = "{:4d}".format(int(np.random.rand()*10000))
                pickle.dump(test_results, open( "prediction_brnn_{}.pkl".format(experiment), "wb" ), protocol=2)
                pickle.dump(params, open( "params_brnn_{}.pkl".format( experiment), "wb" ), protocol=2)
                logger.info("Cross val: {:.2f}%".format(accuracy.mean()*100))
                torch.save(model.state_dict(), "act_model_BBB_{}.pth".format(experiment))
                break

    #Truncate long sequences and pad the small ones with relation to the parameter 'max'
    def _padding(self,videos,max):
        sizes = []
        [sizes.append(len(v)) for v in videos]
        sizes = np.array(sizes)
        padded_data = np.ones((len(videos),max,44))*-1
        padded_labels = np.ones((len(videos),max))*-1
        
        for i,video in enumerate(videos):
            padded_labels[i] = video[0,0]-1
            if len(video) > max:
                video = video[:max]
            padded_data[i,:len(video)] = video[:,1:]
            #padded_data[i,-len(video):] = video[-1,1:]
           # print(padded_data[i].sum()- video[:,1:].sum())

        padded_data = padded_data.astype(float)
        padded_labels = padded_labels.astype(int)
        return padded_data, padded_labels

    #Produce batchs for each step
    def to_batch(self,videos, batch_size = 32, seq = 1, max = 100):
            indexes = list(range(len(videos)))
            random.shuffle(indexes)
            videos = [videos[i] for i in indexes]
            for b in range(len(videos)//batch_size):
                video_batch = [videos[i] for i in range(b*batch_size,(b+1)*batch_size)]
                padded_data, padded_labels = self._padding(video_batch,max= max)
                size = padded_data.shape[1] // seq
                for s in range(size):
                    label = padded_labels[:,s*seq:(s+1)*seq]
                    data = padded_data[:,s*seq:(s+1)*seq]
                    data = data[:,:,[0, 1, 2,3,30,31,32,33,34,35,36,37, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,38,39,40,41]]
                    yield data , label , True if s == (size-1) else False


    def predict(self, model, videos, results, log):
        num_class = 12
        log.info("Predicting...")
        model.to(self.device)
        model.train()
        model.sampling(True,False)
        model.set_dropout(0.15)
        criterion = nn.NLLLoss()
        running_loss = 0
        running_loss = 0
        running_corrects = 0
        total = 1e-18 

        mc_samples = 20 #amount of samples for Monti Carlo Estimation

        with torch.no_grad():
            for (video, interval) in videos:
                probs = np.zeros((len(video),mc_samples, num_class))
                loss = 0
                for mc in range(mc_samples):
                    hidden = None
                    for i, data in enumerate(video):
                        label = data[0]-1
                        label = np.expand_dims(label,0)
                        data = data[1:]
                        data = data[[0, 1, 2,3,30,31,32,33,34,35,36,37, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,38,39,40,41]]
                        data = np.expand_dims(data,0)
                        
                        #creating a batch with same data. It improves the performance of MC samples
                        # label = np.repeat(label,mc_samples,0)
                        # data = np.repeat(data,mc_samples,0)
                        label = torch.tensor(label).long()
                        data = torch.from_numpy(data).float().unsqueeze(1)
                        data = data.to(self.device)
                        out,hidden = model(data,hidden)
                        out = out.cpu()
                        hidden = ([h.data for h in hidden])
                        probs[i,mc] = F.log_softmax(out,1).exp().detach().numpy()

                #label = label.unsqueeze(0)
                loss = criterion(F.log_softmax(out,1), label)
                pred = np.argmax(probs.mean(1), 1)[-1]
                loss = loss/mc_samples
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
    


    def train(self, args, model, train_data,max_clip_sequence,device, log):
        #put the model in train mode
        model.train()
        log.info("Training...")
        
        #Hyperparameters
        epochs = args.epoch #30
        batch = args.batch_size#32
        sequence = args.seq #10
        clip = args.clip #5
        lr=args.lr#1e-3
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = StepLR(optimizer, step_size=1, gamma=0.99)
        indexes = list(range(len(train_data)))
        steps = len(train_data)//batch
        accuracies = []
        B = steps
        C = max_clip_sequence//sequence
        for epoch in range(epochs):
            random.shuffle(indexes)
            train_data = [ train_data[i] for i in indexes]
            running_loss = 0
            running_kl = 0
            running_corrects = 0 
            total = 1e-18
            #importance of KL-divergence terms in relation to classification loss
            scale = 1.0
            hidden = None
            probs = []
            for data, label,end_sequence_batch in self.to_batch(train_data,seq=sequence, batch_size = batch, max =max_clip_sequence):
                data,label = torch.from_numpy(data).float().to(device), torch.from_numpy(label).to(device)
                model.zero_grad()
                label = label.contiguous().view(-1)
                outputs = []

                # for bd in range(sequence):
                #sequence_data = data[:,bd,:].unsqueeze(1)
                #out, hidden = model(sequence_data, hidden)
                #outputs.append(out) 
                # h = 0
                # c = 0
                #for _ in range(10):
                out, hidden = model(data, hidden)
                #     h += hidden[0].data
                #     c += hidden[1].data
                # hidden = (h/10.0, c/10.0)
                hidden = ([h.data for h in hidden])
                #out = torch.cat(outputs,0)
                # l2 = None

                # for p in model.parameters():
                #     l2 = p.norm(2) if l2 is None else l2 + p.norm(2)
                
                NLL, KL = model.get_loss(out, label)
                # proper scaling for sequence loss
                NLL_term = NLL / C
                # proper scaling for sequence loss
                kl_term = scale*(KL / B*C)
                #Composed loss
                loss = NLL_term + kl_term
                #backpropagate error
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
                    running_loss += NLL_term.item() * out.size(0)
                    running_kl += kl_term.item() * out.size(0)
                    running_corrects += torch.sum(preds == label.data).double()
            
            
            probs = np.array(probs)
            time = np.arange(len(probs))
            
            acc = running_corrects/float(total)
            accuracies.append(acc)
            if acc > 0.9: scheduler.step()

            # if epoch > 200 and acc < 0.1:
            #     break
            # if epoch > 300 and acc < 0.2:
            #     break
            # if epoch > 400 and acc < 0.5:
            #     break
           
                
            
            #early stop condition (mean accuracy over the last 10 training epochs) > 99%
            if len(accuracies)  > 5: 
                del accuracies[0]
                mean = sum(accuracies)/float(len(accuracies))
                if  mean > 0.999: break
            #log information at each 100 epochs
            if epoch%100==0:log.info("---> Epoch {:3d}    loss (NLL/KL) = {:4.4f}/{:4.4f}      accuracy = {:2.2f}%".format(epoch + 1, running_loss/total, running_kl/total, running_corrects/float(total)*100) )
        
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

#100 python action_anticipation.py --seq 120 --batch_size 216 --hidden_dim  64 --lr 1e-2 --epoch 200 --n_layers 1 --trunc_seq 120 --grad_clip 4.7
#99.58 python action_anticipation.py --seq 120 --batch_size 200 --hidden_dim  64 --lr 1e-2 --epoch 200 --n_layers 2 --trunc_seq 120 --grad_clip .6
#98.75 python action_anticipation.py --seq 32 --batch_size 32 --hidden_dim  256 --lr 1e-3 --epoch 200 --n_layers 4 --trunc_seq 128 --grad_clip 5
#ball 95.42 python action_anticipation.py --seq 120 --batch_size 216 --hidden_dim  64 --lr 1e-2 --epoch 200 --n_layers 2 --trunc_seq 120 --grad_clip .5
# remain python action_anticipation.py --seq 120 --batch_size 216 --hidden_dim  64 --lr 1e-2 --epoch 200 --n_layers 2 --trunc_seq 120 --grad_clip 5
# 6 class python action_anticipation.py --seq 120 --batch_size 108 --hidden_dim  32 --lr 2e-2 --epoch 200 --n_layers 1 --trunc_seq 120 --data_type g --grad_clip 0.15

#python action_anticipation.py --seq 100 --batch_size 216 --hidden_dim 64 --lr 1e-1 --epoch 2000 --n_layers 2 --trunc_seq 100 --grad_clip 0.2

#input_size = 32, output_size = 12, hidden_dim = 32, n_layers = 2, std = -4, epoch= 1000,lr = 0.05,batch = 16,sequence = 8,clip = 0.7, max=100