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
from  tqdm import tqdm
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

        self.window = []
        self.movement = False
        self.window_size = 3
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

        # self.conv_hand = utils.inflate_conv(features[0],
        #          time_dim=4,
        #          time_padding=0,
        #          time_stride=1,
        #          time_dilation=1,
        #          center=False)
        

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
        self.hand_emb = None
        self.skl_emb = None
        self.batch_size = 1


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


    def predict_spotting(self, hand, skl):
        with torch.no_grad():
            # self.batch_size = hand.shape[0]
            _, _,N,C,H,W = hand.size()
            hand = hand.reshape(1,N,C,H,W)
            hand_l = self.hand_features(hand[:,0])
            hand_r = self.hand_features(hand[:,1])
            skl= skl.reshape(1,-1)
            self.skl_emb = self.skl_embedding(skl)
            hand = self.soft_att([hand_l, hand_r])
            self.hand_emb = self.hand_embedding(hand)
            x = self.soft_att_spt([self.hand_emb, self.skl_emb])
            x = x.view(1, 1,-1)
            if self.batch_size > 1:
                x = x.repeat((self.batch_size,1,1))
            out, hidden = self.rnn_spt(x, self.hidden_spt)
            # self.hidden_spt = ([h.data for h in hidden])
            self.hidden_spt = ([h.data.mean(1,keepdim=True).repeat(1,mc_samples,1) for h in hidden])
            out = self.bfc_output_spt(out) 
            out = out.contiguous().view(-1, out.size(-1))
            probs = F.log_softmax(out,1).exp().cpu().numpy()     
            uncertainty = self._calc_uncertainty(probs)
            mean = probs.mean(0)
            pred = np.argmax(mean)
            prob = mean.max()        
        return pred, prob, probs, uncertainty

    def predict_classsifiy(self):
        with torch.no_grad():
            x = self.soft_att_info([self.hand_emb, self.skl_emb])
            x = x.view(1, 1 ,-1)
            if self.batch_size > 1:
                x = x.repeat((self.batch_size,1,1))
            out, hidden = self.rnn(x, self.hidden)
            # self.hidden = ([h.data for h in hidden])
            self.hidden = ([h.data.mean(1,keepdim=True).repeat(1,self.batch_size,1) for h in hidden])
            out = self.bfc_output(out) 
            out = out.contiguous().view(-1, out.size(-1))
            probs = F.log_softmax(out,1).exp().cpu().numpy()     
            uncertainty = self._calc_uncertainty(probs)
            mean = probs.mean(0)
            pred = np.argmax(mean)
            prob = mean.max()        
        return pred, prob, probs, uncertainty
        
    def _calc_uncertainty(self,probs):
        if len(probs.shape) > 2:
            mean = probs.mean(1)
            h = -(mean*np.log(mean)).sum(1) #entropy
        else: 
            mean = probs.mean(0)
            h = -(mean*np.log(mean)).sum(0) #entropy
            # s = probs.std(0).sum()
        return h
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

    def predict(self, hand, skl):
        # self.set_dropout(0.1)
        spt_pred, spt_prob, spt_probs, spt_unc = self.predict_spotting(hand, skl)
        if len(self.window) == self.window_size: del self.window[0]
        self.window.append(spt_pred)
        w = sum(self.window)/self.window_size


        if w == 1.0 and not self.movement: 
            self.movement = True
        elif w == 0.0 and self.movement: 
            self.movement = False
        
        # print(w, spt_pred, self.movement)
        
        if not self.movement:
            self.hidden = None
            self.soft_att_info.weights = 0
            return 0, spt_prob, spt_probs, spt_unc
        
        pred, prob, probs, uncertainty   = self.predict_classsifiy()
               
        return pred+1, prob, probs, uncertainty      

def crop_center(image, out):
        y,x = image.shape[1], image.shape[2]
        cropy, cropx = out
        startx = x//2-(cropx//2)
        starty = y//2-(cropy//2) 
        startx = 0 if startx + cropx >=x  else startx
        starty = 0 if starty + cropy >=y else starty
        return image[:,starty:starty+cropy,startx:startx+cropx,:]

def _split_image(image):
        if len(image.shape) == 3:
            image = np.expand_dims(image,0)
        image = np.array([[image[:,:50,i*50:(i+1)*50],image[:,50:,i*50:(i+1)*50]] for i in range(4)])
        return image.reshape((-1,50,50,3))

def _unsplit_images(images, output_size):
        h,w = output_size[0], output_size[1]
        images = images.reshape(-1,8, h,w,3)
        image_result = np.zeros(images.shape).reshape(-1, 2, 2*h,2*w,3)
        image_result[:,0,:h,:w] = images[:,0]
        image_result[:,0,:h,w:2*w] = images[:,1]
        image_result[:,0,h:,:w] = images[:,2]
        image_result[:,0,h:,w:2*w] = images[:,3]

        image_result[:,1,:h,:w] = images[:,4]
        image_result[:,1,:h,w:2*w] = images[:,5]
        image_result[:,1,h:,:w] = images[:,6]
        image_result[:,1,h:,w:2*w] = images[:,7]
        return image_result
def data_tranformation(skl,hands, output_size = (40,40)):
    hands = transform.resize(hands, (100,200), preserve_range = True).astype(np.uint8)
    hands = _split_image(hands)
    hands = crop_center(hands,output_size)
    skl = np.array([[skl.centralize().normalize().vectorize_reduced()]])
    skl = torch.from_numpy(skl).float()
    hands = _unsplit_images(hands,output_size).astype(np.float32)/255.0
    hands = np.transpose(hands, (0,1,4,2,3))
    hands = torch.from_numpy(hands).unsqueeze(1)
    return skl, hands


def get_frames(file):
    frames = pims.Video(file)
   
    return frames


def handDetect(skeleton, image, out_size = (240,320)):
    # right hand: wrist 4, elbow 3, shoulder 2
    # left hand: wrist 7, elbow 6, shoulder 5
    images = np.zeros((2,out_size[0],out_size[1],3))
    ratioWristElbow = 0.5
    image_height, image_width = image.shape[0:2]
    
    joints = skeleton.joints

    # if any of three not detected
    
    left_shoulder = skeleton.GetJoint(7)
    left_elbow  = skeleton.GetJoint(8)
    left_wrist = skeleton.GetJoint(9)
    right_shoulder  = skeleton.GetJoint(4)
    right_elbow  = skeleton.GetJoint(5)
    right_wrist = skeleton.GetJoint(6)
    score = 0.6
    has_left = ( (left_shoulder.get3DPoint().sum() !=0 and left_shoulder.score > score) and \
             (left_elbow.get3DPoint().sum() !=0 and left_elbow.score > score) and \
             (left_wrist.get3DPoint().sum() !=0 and left_elbow.score > score) \
             )
    has_right =  ( (right_shoulder.get3DPoint().sum() !=0 and right_shoulder.score > score) and \
             (right_elbow.get3DPoint().sum() !=0 and right_elbow.score > score) and \
             (right_wrist.get3DPoint().sum() !=0 and right_elbow.score > score) \
             )
    
        
    hands = []
    #left hand
    if has_left:
       
        x1, y1, _ = left_shoulder.get3DPoint()
        x2, y2, _ = left_elbow.get3DPoint()
        x3, y3, _ = left_wrist.get3DPoint()
        hands.append([x1, y1, x2, y2, x3, y3, True])
    
    # right hand
    if has_right:
        x1, y1, _ = right_shoulder.get3DPoint()
        x2, y2, _ = right_elbow.get3DPoint()
        x3, y3, _ = right_wrist.get3DPoint()
        hands.append([x1, y1, x2, y2, x3, y3, False])

    for x1, y1, x2, y2, x3, y3, is_left in hands:
        x = x3 + ratioWristElbow * (x3 - x2)
        y = y3 + ratioWristElbow * (y3 - y2)
        distanceWristElbow = math.sqrt((x3 - x2) ** 2 + (y3 - y2) ** 2)
        distanceElbowShoulder = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        width = 1 * max(distanceWristElbow, 0.9 * distanceElbowShoulder)

        x -= width / 2
        y -= width / 2  # width = height
        # overflow the image
        if x < 0: x = 0
        if y < 0: y = 0
        width1 = width
        width2 = width
        if x + width > image_width: width1 = image_width - x
        if y + width > image_height: width2 = image_height - y
        width = min(width1, width2)
        # the min hand box value is 10 pixels
        if width >= 10:
            x =int(x)
            y=int(y)
            w = int(width)
            img = image[y:y+w, x:x+w,:]
            img = m.imresize(img, out_size)
            if is_left: images[0] = img
            else: images[1] = img
       
    return np.uint8(images)

if __name__ == "__main__":
    import time
    import pickle
    mc_samples = 20
    time_pred = []
    time_data = []
    time_transf = []
    # results = {"pred":[], "label":[],  "probs":[]}
    # device = torch.device('cpu')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    dict_save = torch.load("saved_models/model_gesture_spt_8797.00.pth")
    hidden_dim = dict_save["hidden_dim"]
    n_layers = dict_save["num_layers"]
    model = RTGR( 21, 15, hidden_dim,n_layers, mode = "mc", dropout = 0.1)
    model.window_size = 1
    model = model.to(device)
    model.load_state_dict(dict_save["model"],strict= True)
    model.eval()
    model.set_dropout(0.3)
    model.batch_size= mc_samples
    
    files = glob.glob("/notebooks/datasets/ufes-2020-01-23/*3d.json")
    # files = sorted(files,key = lambda  file: file.replace("_3d.json", "").split("/")[-1])
    names = [file.replace("_3d.json", "") for file in files]
    # print(names)
    cams = ["c00", "c01","c02","c03"]
    out_size = (110,120)
    skeletons_3d = []
    count = 0
    result_files = []
    for name,file in tqdm(zip(names,files), total = len(names)):
        # print(name)
        model.hidden_spt = None
        model.soft_att_spt.weights = 0
        videos = []
        files2d = []
        results = {"pred":[], "label": [], "prob":[], "probs":[], "uncertainty":[],  "weigth_spt":[],  "weigth_clf":[]}
        n = name.split("/p")[-1]
        p, g = n.split("g")
        with open(file) as f:
            data = json.load(f)

        label = np.zeros(len(data["localizations"]))
        label_spt = np.zeros(len(data["localizations"]))
        with open(file.replace("_3d","_spots").replace("ufes-2020-01-23","ufes-2020-01-23/spots_left_and_right_hand")) as f:
            spots = json.load(f)
        for l in spots["labels"]:
            b = l["begin"]
            e = l["end"] + 1
            label_spt[b:e] = int(g)
        with open(file.replace("_3d","_spots")) as f:
            spots = json.load(f)
        for l in spots["labels"]:
            b = l["begin"]
            e = l["end"] + 1
            label[b:e] = int(g)
        
        for file in [ "{}{}_2d.json".format(name,cam) for cam in cams]:
            with open(file) as f:
                files2d.append(iter(json.load(f)["annotations"]))
        
        for file in ["{}{}.mp4".format(name,cam) for cam in cams]:
            videos.append(get_frames(file) )

        size = len(videos[0])
        # for sample,  in range(size):

        for sample, localization in enumerate(data["localizations"]):
            annotations = ObjectAnnotations(localization)  
            obj = annotations.objects[0]
            skl_3d = Skeleton(obj)
            skeletons_3d.append(skl_3d)
            hands = [[],[]]
           # t = time.time()
            for i in range(4):
                has_hands = False
                img = videos[i][sample]
                localization = next(files2d[i])
                annotations = ObjectAnnotations(localization) 
                skeletons = [Skeleton(obj) for obj in annotations.objects]
                for skl in skeletons:
                    # joint = skl.GetJoint(10) 
                    # if (joint.x >250 and joint.x < 900) and (joint.y>275 and joint.y<565):
                    hand = handDetect(skl,img, out_size)
                    #hand = [np.uint8(extractSkin(h)) for h in hand]
                    hands[0].append(hand[0])
                    hands[1].append(hand[1])
                    has_hands= True
                    break
                if not has_hands:
                    size = out_size+(3,)
                    hands[0].append(np.zeros(size))
                    hands[1].append(np.zeros(size))
            s = time.time()
            hands = np.vstack([np.hstack(hand) for hand in hands] ).astype(np.uint8)
            time_data.append(time.time()-s)
            s = time.time()
            skl_vec, hands = data_tranformation(skl_3d,hands,(40,40))
            skl_vec = skl_vec.to(device)
            hands = hands.to(device)
            time_transf.append(time.time()-s)
            # print(skl_vec.shape, hands.shape)
            s = time.time()
            pred, prob, probs, uncertainty= model.predict(hands,skl_vec)
            time_pred.append(time.time()-s)
            l,p = (label[sample],pred) if label[sample]>0 else (0,0) if pred>0 and label_spt[sample]>0 else  (label[sample], pred)
            results["pred" ].append(p)
            results["prob" ].append(l)
            results["label"].append(label[sample])
            results["probs"].append(probs)
            results["uncertainty"].append(uncertainty)
            results["weigth_spt"].append(model.soft_att_spt.weights)
            results["weigth_clf"].append(model.soft_att_info.weights)


        

        result_files.append([name,results])
        # break

    corrects = 0
    corrects_spt = 0
    total = 0
    for n,r in result_files:
        p = np.array(r["pred"])
        l = np.array(r["label"])
        corrects += sum(p==l)
        corrects_spt += sum((p>0)==(l>0))
        total += len(p)
    acc = corrects/total
    print(corrects/total, corrects_spt/total)
    print(total)
    time_data = np.array(time_data)
    time_transf = np.array(time_transf)
    time_pred = np.array(time_pred)
    print("Time data {:.4f}/{:.4f}  Time transf {:.4f}/{:.4f}   time pred {:.4f}/{:.4f}".format(time_data.mean(),time_data.std(),time_transf.mean(),time_transf.std(),time_pred.mean(),time_pred.std()))
    pickle.dump(result_files, open("results_splited_{}.pkl".format(int(10000*acc)),"wb"))
    
