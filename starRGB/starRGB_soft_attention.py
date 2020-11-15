!/usr/bin/env python
coding: utf-8


from __future__ import print_function, division
from torch.nn import functional as F
import glob
import os
import torch
import pandas as pd
import Augmentor
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
import cv2
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

#https://github.com/1Konny/gradcam_plus_plus-pytorch/blob/master/gradcam.py
class GradCam:
    def __init__(self, model):
        self.model = model
        self.model.eval()
        

    def forward(self, input, grad_cam_module = 0):
        return self.model(input, grad_cam_module=grad_cam_module)

    def preprocess_image(self, img, normalize = False):
        preprocessed_img = img
        if normalize:
            preprocessed_img = preprocessed_img.copy()[: , :, ::-1]
            means=[0.485, 0.456, 0.406]
            stds=[0.229, 0.224, 0.225]

            for i in range(3):
                preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
                preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
            preprocessed_img = \
                np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
            preprocessed_img = torch.from_numpy(preprocessed_img)

        # preprocessed_img.unsqueeze_(0)
        input = Variable(preprocessed_img, requires_grad = True)
        return input

    def __call__(self, input, class_idx=None, retain_graph=False, grad_cam_module = 0, name="cam"):
        # input = self.preprocess_image(image)

        b, c, h, w = input.size()

        logit = self.model(input, grad_cam_module=grad_cam_module)

        if class_idx == None:
            prob,pred= torch.max(F.log_softmax(logit,1).exp(), 1)
            prob = prob.cpu().data.numpy()[0]
            class_idx = int(pred.cpu().data.numpy()[0])
        else:
            prob = F.log_softmax(logit,1).exp().cpu().data.numpy()[0,class_idx]

       
        score = logit[:, class_idx].squeeze() 
        
        self.model.zero_grad()
        score.backward(retain_graph=retain_graph)
        gradients = self.model.gradients # dS/dA
        activations = self.model.activations # A
        b, k, u, v = gradients.size()

        alpha_num = gradients.pow(2)
        alpha_denom = gradients.pow(2).mul(2) + \
                activations.mul(gradients.pow(3)).view(b, k, u*v).sum(-1, keepdim=True).view(b, k, 1, 1)
        alpha_denom = torch.where(alpha_denom != 0.0, alpha_denom, torch.ones_like(alpha_denom))

        alpha = alpha_num.div(alpha_denom+1e-7)
        positive_gradients = F.relu(score.exp()*gradients) # ReLU(dY/dA) == ReLU(exp(S)*dS/dA))
        weights = (alpha*positive_gradients).view(b, k, u*v).sum(-1).view(b, k, 1, 1)

        saliency_map = (weights*activations).sum(1, keepdim=True)
        saliency_map = F.relu(saliency_map)
        saliency_map = F.upsample(saliency_map, size=(input.size(2), input.size(3)), mode='bilinear', align_corners=False)
        saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
        saliency_map = (saliency_map-saliency_map_min).div(saliency_map_max-saliency_map_min).squeeze().data.numpy()
        
        image = input.squeeze(0).numpy()
        image = np.transpose(image, (1,2,0))
        self.show_cam_on_image(image,saliency_map,name,class_idx,prob)
        return saliency_map, logit

            

    def show_cam_on_image(self, img, mask, name, pred, prob):
        # print(np.max(mask), np.min(mask))
        heatmap = cv2.applyColorMap(np.uint8(255*mask), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255.0
        cam = 0.3*heatmap + 0.7*np.float32(img)
        # cam = cam / np.max(cam)
        cv2.imwrite("{}_pred_{}-{:.2f}.jpg".format(name,pred,prob*100), np.uint8(255 * cam))

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
        if self.data_aug:
            # image = self.crop_center(image, (120,140))
            image = self.randon_crop(image)
            image = self.random_horizontal_flip(image)
            image = self.random_rotation(image)
            image = self.random_noise(image)
            # image = self.random_jitter(image)
        else:
            # image = transforms.Resize((self.output_size[0]+10, self.output_size[1]+10))(image)
            # image = sk.transform.resize(image,list(self.output_size))
            image = self.crop_center(image, self.output_size)
        image = np.transpose(image, (2,0,1))
        image = torch.from_numpy(image.astype(np.float32))/255.0
        # mean = image.view(image.size(0), -1).mean(1)
        # std = image.view(image.size(0), -1).std(1)+ 1e-18
        # image = (image - mean.view(-1,1,1))/std.view(-1,1,1)
        # # print(image.shape)

        return {'image': image, 'label': torch.from_numpy(np.array(label)).long()}
    

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
        if pickle_file == "train":
            # datadict = np.load("./datasets/star_cos_train.npz")
            datadict = self.load_pickle("star_cos_train.pkl")
            # datadict["data"] = np.append(datadict["data"],datadict1["data"],0)
            # datadict["labels"] = np.append(datadict["labels"], datadict1["labels"],0)
            # del datadict1
        else:
            # datadict = np.load("./datasets/star_cos_test.npz")
            datadict = self.load_pickle("star_cos_test.pkl")

        
        self.images, self.labels =  np.array(datadict["data"]), np.array(datadict['labels'])
        print("Labels size {}  min, max = {}-{}".format(self.labels.shape, np.min(self.labels), np.max(self.labels)))
        print("Data size {} min, max = {}-{}".format(self.images.shape, np.min(self.images), np.max(self.images)))
        self.transform = transform
        self.num_classes = 20
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



    dataloaders = {"val":DataLoader(image_datasets["val"], batch_size=batch_size, shuffle=False, num_workers=num_workers),
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
                self.module_list.append(nn.Sequential(*(list(model["model"].children())[:8]), nn.modules.BatchNorm2d(2048))) #resnet

        
        # if len(models) > 1:
        #     self.attention = nn.Sequential(
        #                                 nn.Dropout(p=0.5),
        #                                 nn.Linear(4*4*2048, 32),
        #                                 nn.ReLU(),
        #                                 nn.modules.BatchNorm1d(32),
        #                                 nn.Dropout(p=0.7),
        #                                 nn.Linear(32, 1)
        #                                 )
        #     self.logsoftmax = nn.LogSoftmax(1)


        self.fc = nn.Sequential(nn.Dropout(p=0.5),
                                nn.Linear(2*4*4*2048, 512), 
                                nn.ReLU(), 
                                nn.modules.BatchNorm1d(512),
                                nn.Dropout(p=0.8), 
                                nn.Linear(512, 20)
                                )
                                

        self.features = list(self.module_list.parameters()) #+ list(self.norm.parameters())
        self.gradients = None
        self.activations= None
    
    def freeze_feature(self, freeze = True):
        self.freeze = freeze
        print("Freezing classifier!" if freeze else "Unfreezing classifier!")
        for parameter in self.features:
                parameter.requires_grad = not freeze

    def get_parameters(self):
        classifier = list(self.fc.parameters())
        #if len(self.module_list)>1:
            #classifier += list(self.attention.parameters())
        return self.features, classifier

    def grad_cam_grad(self,grad):
        self.gradients = grad

    def forward(self, x, grad_cam_module = None):
        outputs= []
        att_outputs = []
        self.gradients = []
        # x = self.norm(x)
        if len(self.module_list)>1:

            for i, module in enumerate(self.module_list):
                
                if grad_cam_module == i:   
                    m = x
                    for name, mod in module._modules.items():
                        m = mod(m)
                        if name == "6":
                            m.register_hook(self.grad_cam_grad)
                            self.activations= m
                    
                else:
                    m = module(x)

                outputs.append(m)
                #print(m.shape)
            outputs = torch.stack(list(map(lambda out: out.view(out.size(0),-1), outputs)),1)
            #print(outputs.shape)
            #att_outputs = self.attention(outputs.view(-1, outputs.size(-1)))
            #self.attention_w = self.logsoftmax(att_outputs.view(outputs.size(0),outputs.size(1),1)).exp()
            #outputs  = outputs * self.attention_w
            #x = outputs.mean(1)
            x = outputs.view(outputs.size(0),-1)
            if grad_cam_module == 2:
                x.register_hook(self.grad_cam_grad)
                self.activations= x

        else:
            x = self.module_list[0](x)
        
        x = self.fc(x)
        return x
            
def train_model(model, criterion, optimizers, schedulers, device, dataloaders, num_epochs=25):
    statistics = np.zeros((num_epochs,3))
    since = time.time()
    np.save("statistics.npy",statistics)
    #best_model_wts = copy.deepcopy(model.state_dict())
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
    state = {"acc":best_acc,"state_dict":copy.deepcopy(model.state_dict())}
    for epoch in range(num_epochs):
        for phase in ['train', 'val']:
            if acc_train < 0.95 and phase == "val": continue
            if phase == 'train':
                model.train()  # Set model to training mode
                
                if  acc_train > 0.1:
                    for  scheduler in schedulers:
                        scheduler.step()
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            # Iterate over data.

            for data in dataloaders[phase]:
                inputs, labels = data["image"].to(device), data["label"].to(device)

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    l = None
                    for p in parameters:
                        if l is None:
                            l = p.norm(1) + (p**2).sum()
                        else:
                            l = l+ p.norm(1) + (p**2).sum()
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

        

            if model.module.freeze and acc_train >= 0.1:
                model.module.freeze_feature(False)

            if phase == "train":
                acc_train = epoch_acc 
            else:
                acc_val = epoch_acc
                if epoch_acc > best_acc:
                    after_best = 0
                    best_acc = epoch_acc
                    state = {"acc":best_acc,"state_dict":copy.deepcopy(model.state_dict())}
                    
        statistics[epoch]=np.array([epoch_loss,acc_train,acc_val])
        print(msg.format(epoch, num_epochs - 1, epoch_loss, acc_train*100, acc_val*100, best_acc*100  ), end="")
        after_best += 1
        # if after_best >= 200:
        #     break
    torch.save(state, "./{}.pt".format(file_name))
    np.save("statistics.npy",statistics)
    time_elapsed = time.time() - since
    print('\nTraining complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('\nBest val Acc: {:2f}'.format(best_acc*100))


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

    att = model.module.attention_w.detach().cpu().numpy()
    
    probs = F.log_softmax(output,1).exp()
    probs, classes= probs.max(1)

    return np.squeeze(probs.detach().cpu().numpy()), np.squeeze(classes.cpu().numpy()), np.squeeze(att)
   


def generate_confusion_matrix( predictions, class_names):
        
        def plot_confusion_matrix(cm, classes,
                                    normalize=True,
                                    title='Confusion matrix (%)',
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
               
          
                plt.xticks(tick_marks, classes, rotation=90)
                plt.yticks(tick_marks, classes)

                fmt = '.1f' if normalize else 'd'
                thresh = cm.max() / 2.
                symbol = "%" if normalize else ""
                for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
                    
                    if cm[i, j] > 0:
                        if i == j:
                            plt.text(j, i, format(cm[i, j], fmt),
                                    horizontalalignment="center", fontsize=12,
                                    color="white" if cm[i, j] > thresh else "black")

                plt.tight_layout()
                plt.ylabel('Real')
                plt.xlabel('Predicted')
        Compute confusion matrix
        cnf_matrix = confusion_matrix(predictions["labels"],predictions["classes"])
        np.set_printoptions(precision=2)
        

        # Plot normalized confusion matrix
        plt.grid('on')
        plt.figure(figsize=(13,10))
        plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                            title='Normalized confusion matrix'
                            )
    
        plt.savefig("test.svg", format="svg")
        plt.savefig("./confusion_matrix.png") #Save the confision matrix as a .png figure.
        plt.show()



def load_checkpoint(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location = "cpu")
    print("Acc: ",checkpoint["acc"])
    models_to_ensamble = [
                    {"name":"vgg", "model":models.vgg16_bn(pretrained=False)},
                    {"name":"resnet18", "model":models.resnet50(pretrained=False)}, 
                    {"name":"resnet18", "model":models.resnet101(pretrained=False)}, 
                    ]
    model = Ensemble(models_to_ensamble, name="vgg_resnet")
    model = nn.DataParallel(model)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model
    

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



    loaders = create_datasets(num_workers=8, batch_size=64)
    # info = pd.read_csv("./flower_data/train.csv")[["image","label"]]
    # class_weights = torch.tensor(1.0/info.groupby(["label"]).count().values.astype(np.float32))
    # del info
    models_ensamble = [
                    # {"name":"vgg", "model":models.vgg16_bn(pretrained=True)},
                    {"name":"resnet", "model":models.resnet50(pretrained=True)}, 
                    # {"name":"densenet", "model":models.densenet121(pretrained=True) },
                    {"name":"resnet", "model":models.resnet101(pretrained=True) },
                    ]

    model = Ensemble(models_ensamble, name="star_ensemble")
   

    ft, cl =model.get_parameters()
    model = nn.DataParallel(model)
    model = model.to(device)
    criterion = nn.NLLLoss()
  
    optimizers = [ optim.Adam(ft, lr=1e-4), optim.Adam(cl, lr=5e-3)]
    # # print("")
    # # print('-' * 40)
    # # print("lr = {} bs= {}".format(lr,bs) )
    # # print('-' * 40)

    # # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_schedulers = [lr_scheduler.StepLR(optimizers[0], step_size = 1, gamma = 0.99),
                        lr_scheduler.StepLR(optimizers[1], step_size = 1, gamma = 0.99) ]


    model = [model, criterion, optimizers, exp_lr_schedulers, device]

    model = train_model(*model, loaders, num_epochs = 200)
    





    class_names = ["vattene","vieniqui","perfetto","furbo","cheduepalle","chevuoi","daccordo","seipazzo","combinato","freganiente","ok","cosatifarei","basta","prendere","noncenepiu","fame","tantotempo","buonissimo","messidaccordo","sonostufo"]

    loaders = create_datasets(num_workers=6, batch_size=64)
    model = load_checkpoint('./ensemble_star.pt')
    model = model.to(device)
    # model.to(device)
    grad_cam = GradCam(model = model)
    count = 0
    for data in loaders["val"]:
        inputs, labels = data["image"], data["label"]
        if int(labels.squeeze().numpy()) == 14:
            grad_cam(inputs, class_idx = 14, grad_cam_module = 0, name="cam_test_{}_50_0".format(count)) 
            grad_cam(inputs, class_idx = None, grad_cam_module = 0, name="cam_test_{}_50_1".format(count)) 
            grad_cam(inputs, class_idx = 14, grad_cam_module = 1, name="cam_test_{}_101_0".format(count)) 
            grad_cam(inputs, class_idx = None, grad_cam_module = 1, name="cam_test_{}_101_1".format(count)) 
            grad_cam(inputs, class_idx = 14, grad_cam_module = 2, name="cam_test_{}_ens_0".format(count)) 
            grad_cam(inputs, class_idx = None, grad_cam_module = 2, name="cam_test_{}_ens_1".format(count)) 
            count += 1
            # grad_cam(inputs, grad_cam_module = 1,index = int(labels.squeeze().numpy()), name="cam_test_1") 
    print("total:",count)
    probs, classes, labels, attention = None, None, None,None

    for data in loaders["val"]:
        inputs, label = data["image"], data["label"]
        p, c, att = predict(inputs,model)
        if probs is None:
            probs = p
            classes = c
            labels = label
            attention = att
        else:
            probs = np.concatenate([probs,p])
            classes = np.concatenate([classes,c])
            labels = np.concatenate([labels,label])
            attention = np.concatenate([attention,att])
    
    print(probs.shape, classes.shape, labels.shape, attention.shape)
    np.savez("star_statistics",labels=labels, probs=probs, classes=classes, attention=attention)
    stat = np.load("../results/star_statistics.npz")
    generate_confusion_matrix({"labels":stat["labels"], "classes":stat["classes"]}, class_names)
    print(accuracy_score(stat["labels"], stat["classes"]))


    



