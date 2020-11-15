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
from dataset import IfesDataset
import torch.optim as optim
import matplotlib.pyplot as plt
import utils
import logging
import threading
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable
import bayesian_layers as bl
import random 
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
                                    
        
        self.dropout = dropout
        self.fc = bl.Linear(self.hidden_dim, self.output_size, **last_linear_args)
        self.dp_training = True
        self._baysian_layers = []


    def sampling(self,sampling,store):
        for layer in self._baysian_layers:
            layer.sampling = sampling
            layer.store_samples = store

    def forward(self, x, hidden):
        x = self.fc1(x)
        emb = self.fc2(x)
        out, hidden_out = self.lstm(emb, hidden)
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

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        dataset = IfesDataset()
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
        start = int(clip*10)
        for h in hiddens:
            for l in learning_rates:
                for d in drops:
                #for i in range(start,100):
                    # torch.manual_seed(30)
                    # np.random.seed(30)
                    # random.seed(30) 
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
                    
                    train_data, test_data =  dataset.get_data(spotting = args.spotting)
                    model = self.train(params,model, train_data, max_clip_sequence, self.device, logging)
                    
                    if args.spotting:
                        acc = self.predict_frame_wise(model, test_data,test_results, logging, args.mode)
                    else:
                        acc = self.predict_gesture_wise(model, test_data,test_results, logging, args.mode)
                    
                    if acc > args.thres_train:
                        model_save = {
                            "hidden_dim":h,
                            "num_layers":args.n_layers,
                            "model":model.state_dict()
                        }
                        torch.save(model_save, "saved_models/model_{}_{:.2f}.pth".format(base_name, acc*100))
                        pickle.dump(test_results, open( "predictions/prediction_{}_{:.2f}.pkl".format(base_name,acc*100), "wb" ), protocol=2)


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

        dataset = IfesDataset()
        for _ in range(40):
        #model params
            input_size = 16
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
                model = RTGR("lstm", input_size, output_size, hidden_dim,n_layers, rnn_dropout = 0.5, fc_dropout = 0.5)
                model = model.to(self.device)
                model = self.train(params,model, train_data, max_clip_sequence, self.device, logger)  
                acc = self.predict(model, test_data,test_results, logger)
                accuracy[k] = acc
                if acc < 0.94:break
            
            pickle.dump(test_results, open( "prediction_brnn_{}.pkl".format(experiment), "wb" ), protocol=2)
            if accuracy.mean() > 0.99: 
                experiment = "{:4d}".format(int(np.random.rand()*10000))
                pickle.dump(params, open( "params_brnn_{}.pkl".format( experiment), "wb" ), protocol=2)
                logger.info("Cross val: {:.2f}%".format(accuracy.mean()*100))
                torch.save(model.state_dict(), "act_model_BBB_{}.pth".format(experiment))
                break


#Truncate long sequences and pad the small ones with relation to the parameter 'max'
    def _padding(self,videos,max):
        sizes = []
        [sizes.append(len(v)) for v in videos]
        sizes = np.array(sizes)
        padded_data = np.ones((len(videos),max,21))*-1
        padded_labels = np.ones((len(videos),max))
        
        for i,video in enumerate(videos):

            padded_labels[i] = video[0,0]
            if len(video) > max:
                video = video[:max]
            padded_data[i,:len(video)] = video[:,1:]

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
                    label = padded_labels[:,s*seq:(s+1)*seq]-1
                    data = padded_data[:,s*seq:(s+1)*seq]
                    yield data , label , True if s == (size-1) else False



    #Produce batchs for binary classification
    def to_batch_binary(self,dataset, batch_size = 32, seq = 1, max = 100):

            remind = len(dataset) % max
            shift = random.randint(0, remind)
            indexes = np.arange((len(dataset)//max)-1)*max + shift
            random.shuffle(indexes)
            dataset = np.array([dataset[i:i+max,:] for i in indexes])
                     
            for b in range(len(indexes)//batch_size):
                batch_data = dataset[b*batch_size:(b+1)*batch_size,:,1:]
                batch_label = dataset[b*batch_size:(b+1)*batch_size,:,0].astype(int)
                batch_label[batch_label==-1]=0
                batch_label[batch_label>0]=1
                size = max // seq
                for s in range(size):
                    label = batch_label[:,s*seq:(s+1)*seq]
                    data = batch_data[:,s*seq:(s+1)*seq,:]

                    #repeat batch 100 times
                    #data = np.repeat(data,50,0)
                    #label = np.repeat(label,50,-1)
                    yield data, label , True if s == (size-1) else False

    def predict_gesture_wise(self, model, videos, results, log, mode):
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

        mc_samples = 20 #amount of samples for Monti Carlo Estimation

        with torch.no_grad():
            for (video, interval) in videos:
                probs = np.zeros((len(video),mc_samples, num_class))
                loss = 0
                hidden = None if mode != "bbb" else [None]*mc_samples

                for i, data in enumerate(video):
                    label = data[0]-1
                    label = np.expand_dims(label,0)
                    data = data[1:]
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
        model.set_dropout(args.dropout)

        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = StepLR(optimizer, step_size=1, gamma=0.99)
        steps = len(train_data)//batch
        accuracies = []
        B = steps
        C = max_clip_sequence//sequence
        lengthscale = 3.0
        N = len(train_data)
        weight_decay = lengthscale**2. * (1. - args.dropout) / (2. * N * args.tau)
        
        print("Weight Decay:", weight_decay)

        if args.spotting:
            batch_iter  =  self.to_batch_binary
        else:  
            batch_iter  = self.to_batch
            

        for epoch in range(epochs):
            running_loss = 0
            running_kl = 0
            running_corrects = 0 
            total = 1e-18
            #importance of KL-divergence terms in relation to classification loss
            scale = 1.0
            hidden = None
            probs = []

            
            
            for data, label,end_sequence_batch in batch_iter(train_data,seq=sequence, batch_size = batch, max =max_clip_sequence):
                
                data,label = torch.from_numpy(data).float().to(device), torch.from_numpy(label).to(device)
                model.zero_grad()
                label = label.contiguous().view(-1)
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
                kl_term = scale*(KL / B*C)
                #Composed loss
                loss = NLL_term + kl_term
                #backpropagate error
                loss.backward()
                #clips gradients before update weights
                # nn.utils.clip_grad_norm_(model.parameters(), 0.1)
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

            if acc > 0.6: scheduler.step()

            if epoch > 99 and acc < 0.3:
                break
            # if epoch > 300 and acc < 0.2:
            #     break
            # if epoch > 400 and acc < 0.5:
            #     break
           
                
            #early stop condition (mean accuracy over the last 10 training epochs) > 99%
            # if len(accuracies)  > 5: 
            #     del accuracies[0]
            #     mean = sum(accuracies)/float(len(accuracies))
            #     if  mean > 0.98: break
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

#100 python action_anticipation.py --seq 120 --batch_size 216 --hidden_dim  64 --lr 1e-2 --epoch 200 --n_layers 1 --trunc_seq 120 --grad_clip 4.7
#99.58 python action_anticipation.py --seq 120 --batch_size 200 --hidden_dim  64 --lr 1e-2 --epoch 200 --n_layers 2 --trunc_seq 120 --grad_clip .6
#98.75 python action_anticipation.py --seq 32 --batch_size 32 --hidden_dim  256 --lr 1e-3 --epoch 200 --n_layers 4 --trunc_seq 128 --grad_clip 5
#ball 95.42 python action_anticipation.py --seq 120 --batch_size 216 --hidden_dim  64 --lr 1e-2 --epoch 200 --n_layers 2 --trunc_seq 120 --grad_clip .5
# remain python action_anticipation.py --seq 120 --batch_size 216 --hidden_dim  64 --lr 1e-2 --epoch 200 --n_layers 2 --trunc_seq 120 --grad_clip 5
# 6 class python action_anticipation.py --seq 120 --batch_size 108 --hidden_dim  32 --lr 2e-2 --epoch 200 --n_layers 1 --trunc_seq 120 --data_type g --grad_clip 0.15

#python action_anticipation.py --seq 100 --batch_size 216 --hidden_dim 64 --lr 1e-1 --epoch 2000 --n_layers 2 --trunc_seq 100 --grad_clip 0.2

#input_size = 32, output_size = 12, hidden_dim = 32, n_layers = 2, std = -4, epoch= 1000,lr = 0.05,batch = 16,sequence = 8,clip = 0.7, max=100

# touch gesture2.log && rm -f gesture2.log && python3 RTGR.py --epoch 500 --n_layers 1 --trunc_seq 32 --seq 16 --batch_size 215 --grad_clip 0.1  --mode mc --tau 5 --gesture --logfile gesture2.log --devices 0 --thres_train 0.9
# docker run --runtime=nvidia  -dti -v ~/experiments/:/notebooks -p 60001:8888 --name cabrunco1 clebeson/tftorch:py3