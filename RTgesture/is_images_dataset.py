
import numpy as np
import cv2
import csv
import os
import argparse
import math
import glob
import pickle
import torch
import random
from skeleton import *
import time
from torch.utils.data import Dataset, DataLoader

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
    


class ISImagesDataset(Dataset): 
    def __init__(self, max_seq, type = "train", transform = None):
        self.type = type
        self.max_seq = max_seq
        self.transform = transform
        self.files = glob.glob("hand_images/*.png")
        self.files = sorted(self.files,key = lambda  name:name.replace(".jpg","").split("_")[-1])
        persons = np.array([int(name.split("_")[-3]) for name in self.files])
        if type == "train":
            index = [i for i, p in enumerate(persons) if p not in [17,18]]
        else:
            index = [i for i, p in enumerate(persons) if p in [17,18]]
        
        self.files = self.files[index]
        self.labels =  np.array([[int(name.split("_")[-2]) for name in self.files])
        self.videos = self._segment_videos(self.labels)
        del persons


    def __len__(self):
        return len(self.files)
        
    def __getitem__(self, idx):
        b,e = self.videos[idx]
        files = self.files[b:e]
        labels = torch.tensor([self.labels[b:e]]).long()
        images = torch.tensor([io.imread(img_file) for  img_file in files]).float()/255.0
        sample = {'image': images, 'label': labels}
        if self.transform:
            sample = self.transform(sample)
        return sample  


    def _segment_videos(self, labels):
        labels = labels >0
        before = 0
        
        prev = int(labels[0])
        videos = []
               
        for i,label in enumerate(labels):
            if label != prev: 
                if prev == 0 : before = i
                else: 
                    videos.append([before,i])
                    before = 0
            prev = int(label)
        if before != 0:videos.append([before,i])

        return np.array(videos).astype(int)
       
    def holdout(self, rounds=1, train_percents = 0.8):
        return None      



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