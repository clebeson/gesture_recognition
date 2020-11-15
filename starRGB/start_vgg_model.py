#!/usr/bin/env python
#coding: utf-8


from __future__ import print_function, division
from torch.nn import functional as F
import glob
import os
import torch
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mplimg 
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, models, datasets
import torch.nn as nn
import warnings
import scipy.misc as m
import torch.optim as optim
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR
import time
import copy
import cv2
from PIL import Image
import json
from itertools import product
import pickle
import skimage as sk
import sys  
from torch.autograd import Variable
from sklearn.metrics import confusion_matrix, accuracy_score
warnings.filterwarnings("ignore")
torch.manual_seed(30)
np.random.seed(30)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class DataTrasformation(object):
    def __init__(self, output_size, data_aug = True):
        self.output_size = output_size
        self.data_aug = data_aug
       
  
    def random_jitter(self, image):
        if np.random.rand(1) > 0.8:
            image = transforms.ColorJitter(*(np.random.rand(4)*0.3) )(image)
        return image

    def randon_crop(self, image):
        height, width = self.output_size
        img_height, img_width = (image.shape[0],image.shape[1]) if len(image.shape) <=3 else (image.shape[1],image.shape[2])

        y = 0 if img_height - height <= 0 else np.random.randint(0, img_height - height)
        x = 0 if img_width - width <= 0 else np.random.randint(0, img_width - width)
        assert img_height >= height
        assert img_width >= width
        return image[y:y+height,x:x+width,:]  if len(image.shape) <= 3 else image[:,y:y+height,x:x+width,:] 

    def crop_center(self, img, out):
        y,x = (img.shape[0],img.shape[1]) if len(img.shape) <=3 else (img.shape[1],img.shape[2])
        cropy, cropx = out
        startx = x//2-(cropx//2)
        starty = y//2-(cropy//2) 
        startx = 0 if startx + cropx >=x  else startx
        starty = 0 if starty + cropy >=y else starty
        return img[starty:starty+cropy,startx:startx+cropx,:] if len(img.shape) <= 3 else img[:,starty:starty+cropy,startx:startx+cropx,:]

    def random_rotation(self,image_array):
            if np.random.rand(1) < 0.3:
                random_degree = np.random.uniform(-5, 5)
                if len(image_array.shape) <= 3:
                    image_array = sk.transform.rotate(image_array, random_degree)
                else:
                    for i in range(3):
                        image_array[i] = sk.transform.rotate(image_array[i], random_degree)
            return image_array


    def random_noise(self,image_array):
        if np.random.rand(1) < 0.3:
            image_array = sk.util.random_noise(image_array)
        return image_array
           

    def random_horizontal_flip(self,image_array):
        if np.random.rand(1) < 0.5:
            image_array = image_array[:, ::-1,:] if len(image_array.shape) <= 3 else image_array[:,:, ::-1,:]
        return image_array
           

    
    def __call__(self, sample):
        image, label = sample["image"], sample["label"]
        if self.data_aug:
            image = self.randon_crop(image)
            image = self.random_horizontal_flip(image)
            image = self.random_rotation(image)
            image = self.random_noise(image)
        else:
            image = self.crop_center(image, self.output_size)
        if len(image.shape) <= 3:
            image = np.transpose(image, (2,0,1)).astype(np.float32)
        else:
            image = np.transpose(image, (0,3,1,2)).astype(np.float32)

        image = torch.from_numpy(image)/255.0
        return {'image': image, 'label': torch.from_numpy(np.array(label)).long()}
    


class StarDataset(Dataset):
    """Flower dataset."""

    def __init__(self, dataset = "train", name = "starRGB", transform=None):
        self.files = glob.glob("/notebooks/datasets/Montalbano/{}/images/{}_Sample*.png".format(dataset,name))
        self.transform = transform
        print("{} dataset = {} images".format(dataset,len(self.files)))

    def __len__(self):
        return len(self.files)
  
        
    def __getitem__(self, idx):
        file = self.files[idx]
        image = Image.open(file)
        image = np.asarray(image.resize((120, 160), Image.ANTIALIAS))
        # image = np.load(file)
        # image = np.array([m.imresize(image[i], (120,160)) for i in range(3)])
        sample = {'image': image, 'label': self.get_label(file)}
        if self.transform:
            sample = self.transform(sample)
        return sample

    def get_label(self,name):
        n, _ = os.path.splitext(name)
        n = n.split("_")[-1]
        return int(n) -1 



def create_datasets(num_workers = 2, batch_size = 32,name = "starRGB"):
    image_datasets = {
        "train":StarDataset(dataset="train", name = name,
                               transform=DataTrasformation(output_size=(110,120), data_aug = True)),
        "val":StarDataset( dataset= "validation", name = name,
                              transform=DataTrasformation(output_size=(110,120), data_aug = False)),

        "test":StarDataset( dataset= "test", name = name,
                              transform=DataTrasformation(output_size=(110,120), data_aug = False))
    }
    dataloaders = {
                   "train":DataLoader(image_datasets["train"], batch_size=batch_size,shuffle=True, num_workers=num_workers),
                    "val":DataLoader(image_datasets["val"], batch_size=batch_size, shuffle=False, num_workers=num_workers),
                    "test":DataLoader(image_datasets["test"], batch_size=batch_size, shuffle=False, num_workers=num_workers),
                   }
    return dataloaders


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten,self).__init__()
    def forward(self,x):
        return x.view(x.size(0), -1)

class Star3VGG(nn.Module):
    def __init__(self, name = "image_classifier"):
        super(Star3VGG, self).__init__()
        self.name = name
        self.freeze = False
        vgg1 = models.vgg16(pretrained=True)
        self.vgg1 = nn.Sequential(*(list(vgg1.features.children())[:17]),Flatten())
        vgg2 = models.vgg16(pretrained=True)
        self.vgg2 = nn.Sequential(*(list(vgg2.features.children())[:17]),Flatten())
        vgg3 = models.vgg16(pretrained=True)
        self.vgg3 = nn.Sequential(*(list(vgg3.features.children())[:17]),Flatten())

        self.fc = nn.Sequential(nn.Dropout(p=0.5),
                                nn.Linear(49920, 1024), 
                                nn.ReLU(), 
                                nn.modules.BatchNorm1d(1024),
                                nn.Dropout(p=0.5), 
                                nn.Linear(1024, 20)
                                )
        self.loss_fn = nn.NLLLoss()
                                
        # self.features = list(self.module_list.parameters()) #+ list(self.norm.parameters())
        # self.gradients = None
        # self.activations= None
    
    def freeze_feature(self, freeze = True):
        self.freeze = freeze
        print("Freezing classifier!" if freeze else "Unfreezing classifier!")
        for parameter in self.vgg.parameters():
                parameter.requires_grad = not freeze

    def get_parameters(self):
        features = list(self.vgg1.parameters())
        features += list(self.vgg2.parameters())
        features += list(self.vgg3.parameters())
        return features, self.fc.parameters()
    
    def get_loss(self, output, targets):
        """
        return:
            NLL: Negative Loglikelihood Loss
        """
        return self.loss_fn(F.log_softmax(output,1), targets)


    def forward(self, x, grad_cam_module = None):
        x1 = self.vgg1(x[:,0])
        x2 = self.vgg1(x[:,1])
        x3 = self.vgg1(x[:,2])
        x = (x1+x2+x3)/3
        x = self.fc(x)
        return x

class StarVGG(nn.Module):
    def __init__(self, name = "image_classifier"):
        super(StarVGG, self).__init__()
        self.name = name
        self.freeze = False
        vgg = models.vgg16(pretrained=True)
        self.vgg = nn.Sequential(*(list(vgg.features.children())[:17]),Flatten())

        self.fc = nn.Sequential(nn.Dropout(p=0.5),
                                nn.Linear(49920, 1024), 
                                nn.ReLU(), 
                                nn.modules.BatchNorm1d(1024),
                                nn.Dropout(p=0.5), 
                                nn.Linear(1024, 20)
                                )
        self.loss_fn = nn.NLLLoss()
                                
        # self.features = list(self.module_list.parameters()) #+ list(self.norm.parameters())
        # self.gradients = None
        # self.activations= None
    
    def freeze_feature(self, freeze = True):
        self.freeze = freeze
        print("Freezing classifier!" if freeze else "Unfreezing classifier!")
        for parameter in self.vgg.parameters():
                parameter.requires_grad = not freeze

    def get_parameters(self):
        return self.vgg.parameters(), self.fc.parameters()
    
    def get_loss(self, output, targets):
        """
        return:
            NLL: Negative Loglikelihood Loss
        """
        return self.loss_fn(F.log_softmax(output,1), targets)


    def forward(self, x, grad_cam_module = None):
        x = self.vgg(x)
        x = self.fc(x)
        return x
            
def train_model(model, train_data, val_data, params):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.train()
    epochs = params["epoch"]
    batch = params["batch_size"]
    lr=params["lr"]
    ft, clf = model.get_parameters()

    optimizers = [ optim.Adam(ft, lr=lr[0]), optim.Adam(clf, lr=lr[1])]
    schedulers = [StepLR(optimizers[0], step_size = 1, gamma = 0.99),
                  StepLR(optimizers[1], step_size = 1, gamma = 0.99) ]

    acc_val = 0
    best_acc_val = 0
    accuracies = []
    for epoch in range(epochs):
        total = 0
        running_corrects = 0
        running_loss = 0
        for data in train_data:
            data, label = data["image"].to(device), data["label"].to(device)
            model.zero_grad()
            out = model(data)

            l1 = None
            for p in model.parameters(): 
                l1 = p.norm(1) if l1 is None else l1 + p.norm(1)

            NLL = model.get_loss(out, label)
            loss = NLL   + params["weight_decay"] * l1

            loss.backward()
            optimizers[0].step()
            optimizers[1].step()

            prob, preds = torch.max(out, 1)
            total += out.size(0)
            running_loss += loss.item() * out.size(0)
            running_corrects += torch.sum(preds.data == label.data).double()
        
        
        acc_train = running_corrects/float(total)
        accuracies.append(acc_train)

        if epoch > 30 and acc_train<0.5:break
    
        if acc_train  > 0.5:
            schedulers[0].step()
            schedulers[1].step()
        
        if acc_train > 0.9:
            acc_val = test(model,val_data)

            if acc_val > best_acc_val: 
                best_acc_val = acc_val
                dict_save = {"acc":best_acc_val,"params":params, "model":model.state_dict()}
                torch.save(dict_save, "dynamic_star_rgb_hand_val.pth")

            
        if epoch % 10 == 0:
           print("Epoch = {}  Train ({:.2f}%/{:.2f})  Val ({:.2f}%/{:.2f}%)".format(epoch,acc_train.cpu().numpy()*100, loss.item(), acc_val*100,best_acc_val*100))

        
        #early stop conditions
        if acc_val > 0.99:break
        if len(accuracies)  > 3: 
            del accuracies[0]
            mean = sum(accuracies)/float(len(accuracies))
            if  mean > 0.99: break

    print("Epoch = {}  Train ({:.2f}%/{:.2f})  Val ({:.2f}%/{:.2f}%)".format(epoch,acc_train.cpu().numpy()*100, loss.item(), acc_val*100,best_acc_val*100))

    return model

def test(model, test_data):
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    running_corrects = 0
    total = 0
    with torch.set_grad_enabled(False):
        for data in test_data:
            data, label = data["image"].to(device), data["label"].to(device)
            out = model(data)
            prob, preds = torch.max(out, 1)
            total += out.size(0)
            running_corrects += torch.sum(preds.data == label.data).double()


    acc = running_corrects/float(total) 
    return acc


if __name__ == "__main__":
    model = StarVGG()
    
    for name in ["starcosW","starcos",]:#["starsobel","starsobelW"]: # ["starRGB", "starRGBW", "starRGBdiff", "starRGBdiffW", "star", "starW"]:
        for batch in [96]:
            dataloaders = create_datasets(10,batch,name)
            for lr in [[1e-4,1e-3],[1e-4,1e-4]]:
                print(lr,batch)
                params = {
                    "name":name,
                    "lr":lr,
                    "epoch":200,
                    "batch_size":batch,
                    "weight_decay":1e-5,
                }
                model = train_model(model,dataloaders["train"], dataloaders["val"], params)
                acc= test(model,dataloaders["test"])
                dict_save = {"acc":acc, "params":params, "model":model.state_dict()}
                file_path = "results_vgg/dynamic_star_vgg16_{}_{}.pth".format(name, int(acc*10000))      
                torch.save(dict_save, file_path)  
                print(file_path)
                print("test acc = {}".format(acc*100))

   

    



