from collections import deque
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


class ISDataset(Dataset): 
    def __init__(self, max_seq = 64):
        self.max_seq = max_seq
        self.train = pickle.load(open("ifes_dataset_skl.pkl","rb"))
        self.train_labels = self.train["labels"]
        self.train = self.train["skeletons"]

        self.test = pickle.load(open("ufes_R_dataset_skl.pkl","rb"))
        self.test_labels = self.test["labels"]
        self.test = self.test["skeletons"]
        self.enconded = False
        self.config_data()
       


    def skeleton_augumentation(self,skl):
        
        value = np.random.rand(3)
        if value[0]<0.5:
            skl = skl.add_noise()
        
        if value[1]<0.5:
            skl = skl.flip()
        
        if value[2]<0.5:
            skl = skl.erase_joint()
        return skl


    
    def _encode(self):
        videos = []
        for video, label in self.train:
            # print("{} {} {}".format(len(video), label,len(self.train)))
            video = self._encode_skeletons(video)
            videos.append((video,label))
        self.encoded = True
        self.train = videos
        videos = []
        for video, label, (b,e) in self.test:
            video= self._encode_skeletons(video)
            videos.append((video,label,(b,e)))
        self.test = videos
    
    def get_test(self):
        return self.test

    def get_train(self):
        return self.train


    def _encode_skeletons(self,video, augmentation = False):
        data =  []
        for skl in video:
                if augmentation:
                    skl = self.skeleton_augumentation(skl)
                skl = skl.centralize().normalize()
                vec = skl.vectorize_reduced()
                vec = np.where(vec != -1, (vec-vec.mean())/(vec.std()+1e-18), -1)
        
                # vec = skl.get_normalized_representation()
                # print(vec.sum())
                data.append(vec)     
        data = np.array(data).astype(np.float32)
        return data


    def decode(self, labels):
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
       
      
    
    
    def config_data(self, spotting = False):
    #      self.train = [(t,l) for t,l zip(self.train,self.train_labels)]
    #         self.test  = [(t,l) for t,l zip(self.test,self.test_labels)]
        # else:
        # if not spotting:
        train = self.decode(self.train_labels)
        test = self.decode(self.test_labels)

        indexes_train = list(range(len(train)))
        indexes_test = list(range(len(test)))

        random.shuffle(indexes_train)
        random.shuffle(indexes_test)

        print("train =", len(indexes_train), "  test =", len(indexes_test))
        self.train = [(self.train[b:e],self.train_labels[b]-1) for b,e in train[indexes_train]]
        self.test = [(self.test[b:e],self.test_labels[b]-1,(b,e)) for b,e in test[indexes_test]]
        self._encode()


    def cross_validation(self, k=5):
            #k = 5 if k > 10 or k<1 else k
            fold_size = len(self.train)//k
            indexes = list(range(len(self.train)))
            random.shuffle(indexes)
            for fold in range(k):
                begin = fold*fold_size
                end = begin+fold_size
                if fold == k-1: end = len(self.train)
                val_indexes = indexes[begin:end]
                train = [self.train[index] for index in indexes if index not in val_indexes]
                val = [(self.train[index][0],self.train[index][1],(0,0)) for index in indexes if index in val_indexes]
                yield  train, val



    
