from collections import deque
import numpy as np
import cv2
import csv
import os
import argparse
import math
import glob
import pickle
import random
from skeleton import *
from torch.utils.data import Dataset, DataLoader



class ISDataset(Dataset): 
    def __init__(self, max_seq):
        self.max_seq = max_seq
        self.train = pickle.load(open("ifes_dataset_skl.pkl","rb"))
        self.train_labels = self.train["labels"]
        self.train = self.train["skeletons"]

        self.test = pickle.load(open("ufes_dataset_skl_LR.pkl","rb"))
        self.test_labels = self.test["labels"]
        self.test = self.test["skeletons"]

        self.train_videos, self.train_video_labels, self.test_videos, self.test_video_labels = None, None, None, None

    def __len__(self):
        return len(self.train_video_labels)
        
    def __getitem__(self, idx):
        video, label =  [self.train_videos[idx]], [self.train_video_labels[idx]]
        encoded_video, encoded_label = self._encode_skeletons(video,label, self.max_seq)
        sample = {'skeletons': encoded_video, 'labels': encoded_label}
        return sample    

    def skeleton_augumentation(self,skl):
        
        value = np.random.rand(3)
        if value[0]<0.5:
            skl = skl.add_noise()
        
        if value[1]<0.5:
            skl = skl.flip()
        
        if value[2]<0.5:
            skl = skl.erase_joint()
        return skl
    
    def get_test(self):
        videos = []
        labels =[]
        
        for (video,_), (label,interval) in zip(self.test_videos, self.test_video_labels):
            data = []
            label = np.array(label).astype(int)
            for skl in video:
                skl = skl.centralize().normalize()
                vec = skl.get_normalized_representation()
                data.append(vec)
            videos.append([np.array(data).astype(np.float32), interval])
            labels.append(label)
    

        return videos, labels

    def _encode_skeletons(self,video,labels, max):
        data = []
        labels = np.array([labels[0]]*max).astype(int)
        for i in range(max):
            if i< len(video):
                skl = video[i]
                skl = self.skeleton_augumentation(skl)
                skl = skl.centralize().normalize()
                vec = skl.get_normalized_representation()
                data.append(vec)
            else:
                padding = np.zeros((data[0].shape))-1
                data.append(padding)
        data = np.array(data).astype(np.float32)
        return data, labels

    #Produce batchs for each step
    def to_batch(self, batch_size = 32, seq = 1, max = 100):
            
            indexes = list(range(len(self.train_videos)))
            random.shuffle(indexes)
            videos = [self.train_videos[i] for i in indexes]
            labels = [self.train_video_labels[i] for i in indexes]
            for batch in range(len(videos)//batch_size):
                video_batch = [videos[i] for i in range(batch*batch_size,(batch+1)*batch_size)]
                labels_batch = [labels[i] for i in range(batch*batch_size,(batch+1)*batch_size)]

                encoded_data, encoded_labels = self._encode_skeletons(video_batch,labels_batch,max= max)

                size = encoded_data.shape[1] // seq
                
                for s in range(size):
                    label = encoded_labels[:,s*seq:(s+1)*seq]-1
                    data = encoded_data[:,s*seq:(s+1)*seq]
                    yield data , label , True if s == (size-1) else False



    #Produce batchs for binary classification
    def to_batch_binary(self, batch_size = 32, seq = 1, max = 100):
            self.train_videos
            self.train_video_labels
            self.test_videos
            self.test_video_labels

            remind = len(self.train_video_labels) % max
            shift = random.randint(0, remind)
            indexes = np.arange((len(self.train_video_labels)//max)-1)*max + shift
            random.shuffle(indexes)
            dataset = np.array([self.train_videos[i:i+max,:] for i in indexes])
                     
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

    def _segment_videos(self, labels):
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
    
    
    def config_data(self, spotting = True):
        if spotting:
            return self.train,self.train_labels, self.test, self.test_labels
        else:
            train = self._segment_videos(self.train_labels)
            test = self._segment_videos(self.test_labels)

            indexes_train = list(range(len(train)))
            indexes_test = list(range(len(test)))

            random.shuffle(indexes_train)
            random.shuffle(indexes_test)

            print("train =", len(indexes_train), "  test =", len(indexes_test))
            self.train_videos = [self.train[b:e] for b,e in train[indexes_train]]
            self.train_video_labels = [self.train_labels[b:e] for b,e in train[indexes_train]]
            self.test_videos = [(self.test[b:e],(b,e)) for b,e in test[indexes_test]]
            self.test_video_labels = [(self.test_labels[b:e],(b,e)) for b,e in test[indexes_test]]
            

    def cross_validation(self, k=5, spotting = True):
        if spotting:
            fold_size = len(self.data)//k
            indexes = list(range(len(self.data)))
            data = self.data
        else:
            data = self._segment_videos()
            fold_size = len(data)//k
            indexes = list(range(len(data)))
            random.shuffle(indexes)
        
        print("data size:",len(data))
        
        for fold in range(k):
            begin = fold*fold_size
            end = begin+fold_size
            if fold == k-1: end = len(data)
            val_indexes = indexes[begin:end]
            train_indexes = [index for index in indexes if index not in val_indexes]
            if spotting:
                yield self.data[train_indexes], self.data[val_indexes]
            else:
                
                yield [self.data[b:e] for b,e in data[train_indexes]], [(self.data[b:e],(b,e)) for b,e in data[val_indexes]]



if __name__ == "__main__":
    ActicipateDataset()
    



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