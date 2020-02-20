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


class IfesDataset: 
    def __init__(self):
        # self.data = np.load("ifes_dataset.npy").astype(float)
        # self.data_test = np.load("ufes_dataset_right.npy").astype(float)
        
        self.train = pickle.load(open("ifes_dataset_skl.pkl"))
        self.train = self.train["skeletons"]
        self.train_labels = self.train["labels"]

        self.test = pickle.load(open("ufes_dataset_skl_LR.pkl"))
        self.test = self.test["skeletons"]
        self.test_labels = self.test["labels"]
        # self.data = np.load("extended_train.npy").astype(float)
        #self.data = self._extends_dataset(self.data,3)
       
        # values = self.data[:,1:]
        # min = values.max(1, keepdims = True)
        # values = values - min
        # max = values.max(1, keepdims = True)
        # self.data[:,1:] = (values) / (max+1e-18)

        # values = self.data_test[:,1:]
        # # min = values.max(1, keepdims = True)
        # values = values - min
        # max = values.max(1, keepdims = True)
        # self.data_test[:,1:] = (values) / (max+1e-18)

    def skeleton_augumentation(self,skl):
        value = np.random.rand(3)
        if value[0]<0.5:
            skl = skl.add_noise()
        
        if value[1]<0.5:
            skl = skl.flip()
        
        if value[2]<0.5:
            skl = skl.erase_joint()
        return skl



    def _extends_dataset(self,data,times):
        print("Extending dataset")
        size = len(data)
        aux_data = data
        data = np.zeros([size*int(times)]+list(data.shape[1:]))
        sample = size/float(size*times)
        
        for i in range((size*int(times))-int(times)):
            point = i*sample
            before = int(point)
            rest = point-before
            value = aux_data[before]*(1-rest) + rest*aux_data[before+1]
            label = aux_data[before,0]
            value[value<0] = -1
            value[0] = label
            data[i] = value
        return data


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


    # def _segment_videos(self, data):
    #     b = 0
        
    #     prev = int(data[0,0])
    #     videos = []
               
    #     for i,d in enumerate(data):
    #         if d[0] != prev: 
    #             if prev == 0 : b = i
    #             else: 
    #                 videos.append([b,i])
    #                 b = 0
    #         prev = int(d[0])
    #     if b != 0:videos.append([b,i])

    #     return np.array(videos).astype(int)        
    def holdout(self, rounds=1, train_percents = 0.8):
        return None      
    
    
    def get_data(self, spotting = True):
        if spotting:
            return self., self.data_test
        else:
            train = self._segment_videos(self.data)
            test = self._segment_videos(self.data_test)
            indexes_train = list(range(len(train)))
            indexes_test = list(range(len(test)))
            random.shuffle(indexes_train)
            random.shuffle(indexes_test)
            print("train =", len(indexes_train), "  test =", len(indexes_test))
            return [self.data[b:e] for b,e in train[indexes_train]], [(self.data_test[b:e],(b,e)) for b,e in test[indexes_test]]

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
    



