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
        self.data = np.load("train.npy").astype(float)
        # self.data = np.load("extended_train.npy").astype(float)
        #self.data = self._extends_dataset(self.data,3)

        values = self.data[:,1:]
        mean = values.mean(0, keepdims = True)
        std = values.std(0,  keepdims = True)
        self.data[:,1:] = (values - mean) / (std+1e-18)

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


    def _segment_videos(self):
        b,e = 0,0
        prev = int(self.data[0,0])
        videos = []
               
        for i,d in enumerate(self.data):
            if d[0] != prev: 
                if prev == 0 : b = i
                else: 
                    e = i
                    videos.append([b,e])

            prev = int(d[0])
        return np.array(videos).astype(int)

        
    def holdout(self, rounds=1, train_percents = 0.8):
        return None      

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
    



