#!/usr/bin/env python
# coding: utf-8


from __future__ import print_function, division
from torch.nn import functional as F
import glob
import os
import torch
import pandas as pd
import scipy.misc as m
import argparse
import random
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, models, datasets
import torch.nn as nn
import warnings
import torch.optim as optim
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
import time
import copy
import json
from itertools import product
import pickle
import skimage as sk
import sys  
import sklearn.metrics as metrics
warnings.filterwarnings("ignore")


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
        y = 0 if image.shape[0] - height <= 0 else np.random.randint(0, image.shape[0] - height)
        x = 0 if image.shape[1] - width <= 0 else np.random.randint(0, image.shape[1] - width)
        assert image.shape[1] >= width
        assert image.shape[0] >= height
        return image[y:y+height,x:x+width,:]

    def crop_center(self, img, out):
        y,x = img.shape[0], img.shape[1]
        cropy, cropx = out
        startx = x//2-(cropx//2)
        starty = y//2-(cropy//2) 
        startx = 0 if startx + cropx >=x  else startx
        starty = 0 if starty + cropy >=y else starty

        return img[starty:starty+cropy,startx:startx+cropx,:]

    def random_rotation(self,image_array):
          # pick a random degree of rotation between 25% on the left and 25% on the right
            if np.random.rand(1) < 0.3:
                random_degree = np.random.uniform(-3, 3)
                image_array = sk.transform.rotate(image_array, random_degree)
            return image_array


    def random_noise(self,image_array):
        if np.random.rand(1) < 0.3:
            image_array = sk.util.random_noise(image_array)
        return image_array
           

    def random_horizontal_flip(self,image_array):
          # horizontal flip doesn't need skimage, it's easy as flipping the image array of pixels !
        if np.random.rand(1) < 0.5:
            image_array = image_array[:, ::-1,:]
        return image_array
           

    
    def __call__(self, sample):
        image, label = sample["image"], sample["label"]
        image = sk.transform.resize(image, (110,140))
        if self.data_aug:
            
            # image = self.crop_center(image, (120,140))
            image = self.randon_crop(image)
            # image = self.random_horizontal_flip(image)
            # image = self.random_rotation(image)
            # image = self.random_noise(image)
            # image = self.random_jitter(image)
        else:
            # image = transforms.Resize((self.output_size[0]+10, self.output_size[1]+10))(image)
            # image = sk.transform.resize(image,list((110,140)))
            image = self.crop_center(image, self.output_size)
        image = np.transpose(image, (2,0,1))
        image = torch.from_numpy(image.astype(np.float32))/255.0
        # mean = image.view(image.size(0), -1).mean(1)
        # std = image.view(image.size(0), -1).std(1)+ 1e-18
        # image = (image - mean.view(-1,1,1))/std.view(-1,1,1)
        # # print(image.shape)

        return {'image': image, 'label': torch.from_numpy(np.array(label)).long()}
    
global indices
indices = None
class StarDataset(Dataset):
    """Flower dataset."""

    def __init__(self, pickle_file, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with labels.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        print("Loading ",pickle_file, "numpy dataset...")
        global indices
        datadict = np.load("./star_grit_cos.npz")
        if indices is None or pickle_file == "train": 
            indices = list(range(len(datadict["images"])))
            np.random.shuffle(indices)
        if pickle_file == "train":
            samples = [None for _ in range(9)]
            count = 0
            for img, label in zip(datadict["images"], datadict["labels"]):
                label = int(label)
                if samples[label] is None:
                    samples[label] = img
                    count +=1
                if count == 9: break
            self.images, self.labels =  np.array(datadict["images"][:int(len(indices)*0.8)]), np.array(datadict['labels'][:int(len(indices)*0.8)])

            for i,img in enumerate(samples):
                m.imsave("cos_sample_{}.png".format(i), img)

        else:
            self.images, self.labels =  np.array(datadict["images"][int(len(indices)*0.8):]), np.array(datadict['labels'][int(len(indices)*0.8):])
            # datadict = self.load_pickle("./datasets/star_cos_test.pkl")

        
        print("Labels size {}  min, max = {}-{}".format(self.labels.shape, np.min(self.labels), np.max(self.labels)))
        print("Data size {} min, max = {}-{}".format(self.images.shape, np.min(self.images), np.max(self.images)))
        self.transform = transform
        self.num_classes = 9
        self.identity = np.eye(self.num_classes)
        self.cat_to_name = None
        # with open('cat_to_name.json', 'r') as f:
        #     self.cat_to_name = json.load(f)

    def load_pickle(self, pickle_file):
        try:
            with open(pickle_file, 'rb') as f:
                pickle_data = pickle.load(f)
        except Exception as e:
            with open(pickle_file, 'rb') as f:
                pickle_data = pickle.load(f, encoding='latin1')
        except Exception as e:
            print('Unable to load data ', pickle_file, ':', e)
            raise
        return pickle_data

    def __len__(self):
        return len(self.labels)
    
    def get_class_weights(self):
        weights = 1.0/self.info.groupby(["label"]).count()
        return np.ones(20)*(1.0/20)
        
    def number_of_classes(self):
        return self.num_classes
    
    def class_names(self):
        return self.cat_to_name
    
    def one_hot(self, label):
        return self.identity[label]
        
    def __getitem__(self, idx):
        images, label =  self.images[idx], self.labels[idx]

        sample = {'image': images, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample    


def create_datasets(num_workers = 2, batch_size = 32):

    image_datasets = {
        "train":StarDataset(pickle_file="train",
                               transform=DataTrasformation(output_size=(110,120), data_aug = True)),
        "val":StarDataset( pickle_file= "val",
                              transform=DataTrasformation(output_size=(110,120), data_aug = False))
    }



    dataloaders = {"val":DataLoader(image_datasets["val"], batch_size=200, shuffle=False, num_workers=num_workers),
                   "train":DataLoader(image_datasets["train"], batch_size=batch_size,shuffle=True, num_workers=num_workers)}
    return dataloaders


class Ensemble(nn.Module):
    def __init__(self, models, name = "image_classifier"):
        super(Ensemble, self).__init__()
        self.name = name
        self.freeze = False
        self.module_list = nn.ModuleList()
        # self.norm = nn.modules.BatchNorm2d(3)
        for model in models:
            if  model["name"] == "densenet":
                self.module_list.append(nn.Sequential(*(list(model["model"].features.children())[:9]), nn.modules.BatchNorm2d(1024), nn.AvgPool2d(kernel_size = [7,8]))) #resnet
            else:   
                self.module_list.append(nn.Sequential(*(list(model["model"].children())[:8]), nn.modules.BatchNorm2d(2048),  nn.AvgPool2d(kernel_size = [4,4])    )) #resnet

        if len(models) > 1:
            self.attention = nn.Sequential(nn.Linear(2048, 128),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.7),
                                        nn.modules.BatchNorm1d(128),
                                        nn.Linear(128, 1)
                                        )
                                        
            self.logsoftmax = nn.LogSoftmax(1)


        self.fc = nn.Sequential(nn.Linear(2048, 1024), 
                                nn.ReLU(), nn.Dropout(p=0.8), 
                                nn.modules.BatchNorm1d(1024),
                                nn.Linear(1024, 9)
                                )
                                

        self.features = list(self.module_list.parameters()) #+ list(self.norm.parameters())
    
    def freeze_feature(self, freeze = True):
        self.freeze = freeze
        print("Freezing classifier!" if freeze else "Unfreezing classifier!")
        for parameter in self.features:
                parameter.requires_grad = not freeze

    def get_parameters(self):
        classifier = list(self.fc.parameters())
        if len(self.module_list)>1:
            classifier += list(self.attention.parameters())
        return self.features, classifier

                    
    def forward(self, x):
        outputs= []
        att_outputs = []
        # x = self.norm(x)
        if len(self.module_list)>1:
            for module in self.module_list:
                m = module(x)
                outputs.append(m.view(m.size(0), -1))
                att_outputs.append(self.attention(outputs[-1]))
       
            att_outputs = torch.cat(att_outputs,1)
            self.attention_w = self.logsoftmax(att_outputs).exp().unsqueeze(-1)
            
            outputs = torch.cat(tuple(map(lambda out: out.unsqueeze(1), outputs)),1)
            outputs *= self.attention_w
            x = torch.sum(outputs,1)
        else:
            x = self.module_list[0](x)
            
        x = x.view(x.size(0),-1)
        x = self.fc(x)
        return x
            
    

def train_model(model, criterion, optimizers, schedulers, device, dataloaders, num_epochs=25):
    statistics = np.zeros((num_epochs,3))
    since = time.time()
    np.save("statistics.npy",statistics)
    best_model_wts = copy.deepcopy(model.module.state_dict())
    best_acc = 0.0
    # freeze = True
    model.module.freeze_feature()
    after_best = 0
    acc_train = 0
    acc_val = 0
    epoch_acc = 0
    ft,cl = model.module.get_parameters()
    parameters = list(ft)+list(cl)
    msg = "\r\tEpoch: {}/{}  Epoch loss : {:.4f}   Train Acc : {:.2f}%   Val Acc : {:.2f}%    Best Acc : {:.2f}%    "
    file_name = "ensemble_star"
    state = {"acc":best_acc,"state_dict":model.module.state_dict()}
    best_model = copy.deepcopy(model.module)
    for epoch in range(num_epochs):
        for phase in ['train', 'val']:
            if acc_val == 0 and phase == "val"  and acc_train < 0.95  : continue
            if phase == 'train':
                for  scheduler in schedulers:
                    scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            # Iterate over data.

            for data in dataloaders[phase]:
                inputs, labels = data["image"].to(device), data["label"].to(device)
               
                # zero the parameter gradients
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    l = None
                    for p in parameters:
                        if l is None:
                            l = p.norm(1)
                        else:
                            l = l+p.norm(1)
                    loss = criterion(F.log_softmax(outputs,1), labels) + l*1e-5

                    # backward + optimize only if in training phase
                    if phase == 'train':
                                             
                        # for  scheduler in schedulers:
                        #     scheduler.step()

                        for optimizer in optimizers:
                            optimizer.zero_grad()

                        loss.backward()
                        for optimizer in  optimizers:
                            optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            # print('{} Loss: {:.4f} Acc: {:.2f}'.format(
                # phase, epoch_loss, epoch_acc*100))

           
            
            if model.module.freeze and acc_train >= 0.1:
                print("")
                model.module.freeze_feature(False)

            if phase == "train":
                acc_train = epoch_acc 
            else:
                acc_val = epoch_acc
                if epoch_acc > best_acc:
                    after_best = 0
                    best_acc = epoch_acc
                    state = {"acc":best_acc,"state_dict":model.module.state_dict()}
                    best_model = copy.deepcopy(model.module)
                    torch.save(state, "./{}.pt".format(file_name))
        statistics[epoch]=np.array([epoch_loss,acc_train,acc_val])
        print(msg.format(epoch, num_epochs - 1, epoch_loss, acc_train*100, acc_val*100, best_acc*100  ), end="")
        after_best += 1
        # if after_best >= 200:
        #     break
    np.save("statistics.npy",statistics)
    time_elapsed = time.time() - since
    print('\nTraining complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('\nBest val Acc: {:2f}'.format(best_acc*100))
    return best_model


def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax



def predict(image, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    

    # img = DataTrasformation((110,120),False)(img)
    image= image.to(device)
    # img=torch.unsqueeze(img,0)
    output = model(image)
    
    probs = nn.Softmax(1)(output)
    probs, classes= probs.max(1)

    return np.squeeze(probs.detach().cpu().numpy()), np.squeeze(classes.cpu().numpy())
   





def generate_confusion_matrix( predictions, class_names):
        
        def plot_confusion_matrix(cm, classes,
                                    normalize=False,
                                    title='Confusion matrix',
                                    cmap=plt.cm.Blues):
                """
                This function prints and plots the confusion matrix.
                Normalization can be applied by setting `normalize=True`.
                """
                if normalize:
                    cm = 100 * cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                    print("Normalized confusion matrix")
                else:
                    print('Confusion matrix, without normalization')

                print(np.diag(cm))

                plt.imshow(cm, interpolation='nearest', cmap=cmap)
                plt.title(title)
                plt.colorbar()
                
                tick_marks = np.arange(len(classes))
               
          
                plt.xticks(tick_marks, classes, rotation=45)
                plt.yticks(tick_marks, classes)

                fmt = '.2f' if normalize else 'd'
                thresh = cm.max() / 2.
                symbol = "%" if normalize else ""
                for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
                    plt.text(j, i, format(cm[i, j], fmt)+symbol,
                            horizontalalignment="center",
                            color="white" if cm[i, j] > thresh else "black")

                plt.tight_layout()
                plt.ylabel('Real')
                plt.xlabel('Predicted')
        # Compute confusion matrix
        cnf_matrix = metrics.confusion_matrix(predictions["labels"],predictions["classes"])
        np.set_printoptions(precision=2)
        

        # # Plot normalized confusion matrix
        plt.figure(figsize=(10,10))
        plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                            title='Normalized confusion matrix')
        plt.grid('off')

        plt.savefig("./confusion_matrix.png") #Save the confision matrix as a .png figure.
        plt.show()



def load_checkpoint(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location = "cpu")
    print("Acc: ",checkpoint["acc"])
    models_to_ensamble = [
                    # {"name":"vgg", "model":models.vgg16_bn(pretrained=False)},
                    {"name":"resnet18", "model":models.resnet50(pretrained=False)}, 
                    {"name":"resnet18", "model":models.resnet101(pretrained=False)}, 
                    ]
    model = Ensemble(models_to_ensamble, name="vgg_resnet")
    # model = nn.DataParallel(model)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model
    
def parse_args():
    desc = "Tensorflow implementation of ResNet"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--phase', type=str, default='train', help='train or test ?')
    # parser.add_argument('--dataset', type=str, default='tiny', help='[cifar10, cifar100, mnist, fashion-mnist, tiny')


    parser.add_argument('--epoch', type=int, default=300, help='The number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=128, help='The size of batch')
    # parser.add_argument('--res_n', type=int, default=18, help='18, 34, 50, 101, 152')

    parser.add_argument('--lrf', type=float, default=1e-4, help='learning rates feture')
    parser.add_argument('--lrc', type=float, default=1e-3, help='learning rates classifier')


    # parser.add_argument('--checkpoint_dir', type=str, default='checkpoint',
    #                     help='Directory name to save the checkpoints')
    # parser.add_argument('--log_dir', type=str, default='logs',
    #                     help='Directory name to save training logs')

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    acc, recall, precision, f1 = [], [], [], []
    for i in range(5):
        loaders = create_datasets(num_workers=6, batch_size=args.batch_size)
        # info = pd.read_csv("./flower_data/train.csv")[["image","label"]]
        # class_weights = torch.tensor(1.0/info.groupby(["label"]).count().values.astype(np.float32))
        # del info
        models_ensamble = [
                        # {"name":"vgg", "model":models.vgg16_bn(pretrained=True)},
                        # {"name":"resnet18", "model":models.resnet50(pretrained=True)}, 
                        {"name":"resnet18", "model":models.resnet50(pretrained=True)}, 
                        {"name":"resnet18", "model":models.resnet101(pretrained=True)}, 
                        ]

        model = Ensemble(models_ensamble, name="star_ensemble")
        ft, cl =model.get_parameters()
        model = nn.DataParallel(model)
        model = model.to(device)
        criterion = nn.NLLLoss()
    

        optimizers = [ optim.Adam(ft, lr=args.lrf), optim.Adam(cl, lr=args.lrc)]
        # # print("")
        # # print('-' * 40)
        # # print("lr = {} bs= {}".format(lr,bs) )
        # # print('-' * 40)

        # # Decay LR by a factor of 0.1 every 7 epochs
        exp_lr_schedulers = [lr_scheduler.StepLR(optimizers[0], step_size = 1, gamma = 0.99),
                            lr_scheduler.StepLR(optimizers[1], step_size = 1, gamma = 0.99) ]


        model = [model, criterion, optimizers, exp_lr_schedulers, device]

        model = train_model(*model, loaders, num_epochs = args.epoch)
    



        # class_names = ['prendere', 'vieniqui', 'perfetto', 'fame', 'sonostufo', 'seipazzo', 'basta', 'cheduepalle', 'noncenepiu', 'chevuoi',
        #             'ok', 'combinato', 'freganiente', 'cosatifarei', 'buonissimo', 'vattene', 'messidaccordo', 'daccordo', 'furbo', 'tantotempo']

        # loaders = create_datasets(num_workers=6, batch_size=100)
        # model = load_checkpoint('./ensemble_star_94_58.pt')
        probs, classes, labels = None, None, None

        for data in loaders["val"]:
            inputs, label = data["image"], data["label"]
            p, c = predict(inputs,model)
            if probs is None:
                probs=p
                classes = c
                labels = label.numpy()
            else:
                probs = np.concatenate([probs,p])
                classes = np.concatenate([classes,c])
                labels = np.concatenate([labels, label.numpy()])
        np.savez("star_statistics_{}".format(i),labels=labels, probs=probs, classes=classes)
        # generate_confusion_matrix({"labels":labels, "classes":classes}, class_names)
        acc.append(metrics.accuracy_score(labels, classes))
        recall.append(metrics.recall_score(labels, classes, average="macro"))
        precision.append(metrics.precision_score(labels, classes, average="macro"))
        f1.append(metrics.f1_score(labels, classes, average="macro"))

acc = np.array(acc)*100
recall = np.array(recall)*100
f1 = np.array(f1)*100
precision = np.array(precision)*100
print(acc,recall,f1,precision)

print("Acc: {:.2f}+-{:.2f}".format(np.mean(acc),np.std(acc)))
print("Precision: {:.2f}+-{:.2f}".format(np.mean(precision),np.std(precision)))
print("Recall: {:.2f}+-{:.2f}".format(np.mean(recall),np.std(recall)))
print("F1-score: {:.2f}+-{:.2f}".format(np.mean(f1),np.std(f1)))
        


    


