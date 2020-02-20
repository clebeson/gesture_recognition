
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
from parallel import DataParallelModel, DataParallelCriterion
warnings.filterwarnings("ignore")
torch.manual_seed(30)
np.random.seed(30)


weight_train = []
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
                random_degree = np.random.uniform(-5, 5)
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
            image = sk.transform.resize(image,(120,140))
            # image = self.crop_center(image, (120,140))
            image = self.randon_crop(image)
            image = self.random_horizontal_flip(image)
            image = self.random_rotation(image)
            image = self.random_noise(image)
            # image = self.random_jitter(image)
            image = torch.from_numpy(image.astype(np.float32))
            image = transforms.RandomErasing()(image)
            image = image.permute(2,0,1)/255.0
            # return {'image': image, 'label': torch.from_numpy(np.array(label)).long()}

        else:
            image = sk.transform.resize(image,(120,140))
 #           image = self.crop_center(image, self.output_size)
            #image = sk.transform.resize(image,list(self.output_size))
            # image = transforms.Resize((self.output_size[0], self.output_size[1]))(image)
            image = Image.fromarray(np.uint8(image))
#            image = np.transpose(image, (2,0,1)).astype(np.float32)
            #image = torch.from_numpy(image.astype(np.float32))/255.0
            crops = transforms.FiveCrop(list(self.output_size))(image)
            image = torch.stack([transforms.ToTensor()(crop)/255.0 for crop in crops], 0)

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
            datadict = self.load_pickle("/notebooks/datasets/isoGD/train/starRGB/train_complete.pkl")
            datadict_valid = self.load_pickle("/notebooks/datasets/isoGD/valid/starRGB/valid.pkl")
            datadict["data"] = np.append(datadict["data"],datadict_valid["data"],0)
            datadict["labels"] = np.append(datadict["labels"], datadict_valid["labels"],0)
            del datadict_valid
            _,c = np.unique(np.array(datadict['labels'])-1, return_counts=True)
            w = c/c.min()
            weight_train.append(w.astype(np.float32))

        elif pickle_file == "val":
            datadict = self.load_pickle("/notebooks/datasets/isoGD/valid/starRGB/valid.pkl")
        else:
            # datadict = np.load("./datasets/star_cos_test.npz")
            datadict = self.load_pickle("/notebooks/datasets/isoGD/test/starRGB/test.pkl")

        
        self.images, self.labels =  np.array(datadict["data"]), np.array(datadict['labels'])-1
        

        print("Labels size {}  min, max = {}-{}".format(self.labels.shape, np.min(self.labels), np.max(self.labels)))
        print("Data size {} min, max = {}-{}".format(self.images.shape, np.min(self.images), np.max(self.images)))
        self.transform = transform
        self.num_classes = 249
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
        # "val":StarDataset( pickle_file= "val",
                            #   transform=DataTrasformation(output_size=(110,120), data_aug = False)),
        "test":StarDataset( pickle_file= "test",
                              transform=DataTrasformation(output_size=(110,120), data_aug = False))
    }



    dataloaders = {
        # "val":DataLoader(image_datasets["val"], batch_size=batch_size, shuffle=False, num_workers=num_workers),
                   "test":DataLoader(image_datasets["test"], batch_size=120, shuffle=False, num_workers=num_workers),
                   "train":DataLoader(image_datasets["train"], batch_size=batch_size,shuffle=True, num_workers=num_workers)}
    
    return dataloaders


class Ensemble(nn.Module):
    def __init__(self, models, name = "starrgb_isoGD"):
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

       
        self.fc = nn.Sequential(nn.Linear(2048, 3072), 
                                nn.ReLU(), nn.Dropout(p=0.85), 
                                nn.modules.BatchNorm1d(3072),
                                nn.Linear(3072, 249)
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
        return F.log_softmax(x,1)
            
            
def train_model(model, criterion, optimizers, schedulers, device, dataloaders, num_epochs=25):
    statistics = np.zeros((num_epochs,3))
    since = time.time()
    np.save("statistics.npy",statistics)
    #best_model_wts = copy.deepcopy(model.state_dict())
    best_acc_test = 0.0
    best_acc_val = 0.0
    # freeze = True
    model.module.freeze_feature(False)
    #model.freeze_feature()
    after_best = 0
    acc_train = 0
    acc_val = 0
    acc_test = 0
    epoch_acc = 0
    ft,cl = model.module.get_parameters()
    #ft,cl = model.get_parameters()
    parameters = list(ft)+list(cl)
    msg = "\r\tEpoch: {}/{}  Epoch loss : {:.4f}   Train Acc : {:.2f}%   Val Acc/Best acc : {:.2f}% / {:.2f}%   Test Acc/Best acc : {:.2f}% / {:.2f}%  "
    file_name = "ensemble_star"
    state = {"acc":best_acc_test,"state_dict":copy.deepcopy(model.module.state_dict())}
    for epoch in range(num_epochs):
        for phase in ['train', 'test']:
           
            if acc_train < 0.4 and phase != "train": continue
            
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
                # zero the parameter gradients
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    if phase != "train":
                        shape = list(inputs.shape)
                        inputs= inputs.view(([-1]+shape[2:]))
                        outputs = model(inputs)
                        outputs = torch.cat(outputs,0)
                        outputs = outputs.view((shape[:2]+[-1])).mean(1)
                        
                    else:
                        outputs = model(inputs)
                        l = None
                        for p in parameters:
                            if l is None:
                                l = p.norm(1) + (p**2).sum()
                            else:
                                l = l+ p.norm(1) + (p**2).sum()
                        loss = criterion(outputs, labels)  + l*5e-5
                        outputs = torch.cat(outputs,0)
                   
                    _, preds = torch.max(outputs, 1)


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

            if model.module.freeze and acc_train >= 0.5:
                model.module.freeze_feature(False)

            if phase == "train":
                acc_train = epoch_acc 
            elif phase == "val":
                acc_val = epoch_acc
                if epoch_acc > best_acc_val:
                    after_best = 0
                    best_acc_val = epoch_acc
            else:
                acc_test = epoch_acc
                if epoch_acc > best_acc_test:
                    after_best = 0
                    best_acc_test = epoch_acc
                    state = {"acc":best_acc_test,"state_dict":copy.deepcopy(model.module.state_dict())}
                 
                    
        statistics[epoch]=np.array([epoch_loss,acc_train,acc_val])
        print(msg.format(epoch, num_epochs - 1, epoch_loss, acc_train*100, acc_val*100, best_acc_val*100, acc_test*100, best_acc_test*100  ), end="")
        after_best += 1
        # if after_best >= 200:
        #     break
    torch.save(state, "./{}.pt".format(file_name))
    np.save("statistics.npy",statistics)
    time_elapsed = time.time() - since
    print('\nTraining complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('\nBest Acc (test/val): {:2f}/{:2f}'.format(best_acc_test*100, best_acc_val*100))


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

    #att = model.attention_w.detach().cpu().numpy()
    
    probs = output.exp()
    probs, classes= probs.max(1)

    return np.squeeze(probs.detach().cpu().numpy()), np.squeeze(classes.cpu().numpy()) #, np.squeeze(att)
   


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
        #Compute confusion matrix
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
    checkpoint = torch.load(checkpoint_path, map_location = "cpu")#, map_location = "cpu"
    print("Acc: ",checkpoint["acc"])
    models_to_ensamble = [
                    {"name":"resnet", "model":models.resnet50(pretrained=False)}, 
                    {"name":"resnet", "model":models.resnet101(pretrained=False)}, 
                    ]
    model = Ensemble(models_to_ensamble, name="star_ensemble")
    model.load_state_dict(checkpoint["state_dict"])
    # model = nn.DataParallel(model)
    model.eval()
    return model

def train():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders = create_datasets(num_workers=32, batch_size=600)
    # info = pd.read_csv("./flower_data/train.csv")[["image","label"]]
    # class_weights = torch.tensor(1.0/info.groupby(["label"]).count().values.astype(np.float32))
    # del info
    models_ensamble = [
                    # {"name":"vgg", "model":models.vgg16_bn(pretrained=True)},
                    {"name":"resnet", "model":models.resnet50(pretrained=True)}, 
                    # {"name":"densenet", "model":models.densenet121(pretrained=True) },
                    {"name":"resnet", "model":models.resnet101(pretrained=True) },
                    ]

    # model = Ensemble(models_ensamble, name="star_ensemble")
    model = load_checkpoint("ensemble_iso_star_5118.pt")

    ft, cl =model.get_parameters()
    # model = nn.DataParallel(model)
    model = DataParallelModel(model)
    model = model.to(device)
    weight = torch.from_numpy(weight_train[0]).to(device)
    criterion = nn.NLLLoss(weight)
    criterion = DataParallelCriterion(criterion)
  
    optimizers = [ optim.Adam(ft, lr=5e-4), optim.Adam(cl, lr=5e-3)]
    # # print("")
    # # print('-' * 40)
    # # print("lr = {} bs= {}".format(lr,bs) )
    # # print('-' * 40)

    # # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_schedulers = [lr_scheduler.StepLR(optimizers[0], step_size = 1, gamma = 0.995),
                        lr_scheduler.StepLR(optimizers[1], step_size = 1, gamma = 0.992) ]


    model = [model, criterion, optimizers, exp_lr_schedulers, device]

    model = train_model(*model, loaders, num_epochs = 100)



def test():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data = np.load("/notebooks/datasets/UTD/test/starRGB/starRGB.npz")
    model = load_checkpoint("ensemble_iso_star.pt").to(device)
    images, names = data["images"],data["labels"]
    file1 = open("test_list.txt","w") 
    L = []  


    for img, name in zip(images,names):
        img = sk.transform.resize(img, [110,120])
        img = np.transpose(img,(2,0,1))
        img = torch.from_numpy(img.astype(np.float32)).unsqueeze(0)
        _, pred = predict(img,model)
        name = "test/{folder}/M_{id}.avi test/{folder}/K_{id}.avi {label} \n".format(folder = name.split("_")[0], id = name.split("_")[-1], label = pred+1)
        L.append(name)
        print(name)
    file1.writelines(L) 
    file1.close() #to change file access modes 

if __name__ == "__main__":
   # test()
   train()
    
    



    



