from collections import deque
import numpy as np
import cv2
import csv
import os
import argparse
import math
import pandas as pd
import glob
import pickle
import random


class ActicipateDataset:

    def __init__(self):
        self.data = np.load("labels_normalized.npy").astype(float)
        self._decode_data()
        values = self.data[:,2:]
        #values[values<0] = 0
        mean = values.mean(0, keepdims = True)
        std = values.std(0,  keepdims = True)
        self.data[:,2:] = (values - mean) / (std+1e-18)

    
    def _decode_data(self):
        #normalizing points frame size 640x480
        #self.data[5423:5443,1] = 9
        #self.data[14408:14416,1] = 3
        
        
        # size = np.array([[[640.,480.]]])
        # values = self.data[:,2:].reshape((-1,22,2)) #the two first values correponds to frame_id and label
        # values /= size
        # self.data[:,2:]= np.where(self.data[:,2:] <0, -1,values.reshape((-1,44)))
        
        
        
        
        # print(self.data[:,2:].min(),self.data[:,2:].max())
        # df = pd.DataFrame(columns = ["frame_id","label","ball_x","ball_y","joint_0_x","joint_0_y","joint_1_x","joint_1_y","joint_2_x","joint_2_y","joint_3_x","joint_3_y","joint_4_x","joint_4_y","joint_5_x","joint_5_y","joint_6_x","joint_6_y","joint_7_x","joint_7_y","joint_8_x","joint_8_y","joint_9_x","joint_9_y","joint_10_x","joint_10_y","joint_11_x","joint_11_y","joint_12_x","joint_12_y","joint_13_x","joint_13_y","joint_14_x","joint_14_y","joint_15_x","joint_15_y","joint_16_x","joint_16_y","joint_17_x","joint_17_y","hand_right_x","hand_right_y","hand_left_x","hand_left_y","gaze_fixation_x","gaze_fixation_y"], data= self.data)
        # df.frame_id = df.frame_id.astype(int)
        # df.set_index("frame_id")
        # df.to_csv("labels_cut.csv")
        # np.save("labels_normalized.npy",self.data)
        i = -1
        b,e = 0,0
        prev = int(self.data[0,1])
        videos = []
        count = 0
        
        for d in self.data:
            if d[1] != prev:
                e = int(d[0]-1)
                #print("{}-{}-{} - {}".format(prev,b,e, e-b))
                #if prev not in [7,8,9,10,11,12]:
                videos.append([b,e])
                prev = int(d[1])
                b = int(d[0])
                count += 1
        e = int(d[0])
        #if prev not in [7,8,9,10,11,12]:
        videos.append([b,e])
        self.data = self.data.astype(float)
        self.videos = np.array(videos).astype(int)
        print(len(videos))

    def holdout(self, rounds=1, train_percents = 0.8):
        rounds=1 if rounds < 1 else rounds
        train_percents = 0.8 if train_percents>1 or train_percents <0 else train_percents
        indexes = list(range(len(self.videos)))
        for round in range(rounds):
            random.shuffle(indexes)
            cut = int(train_percents*240)
            train_indexes = indexes[:cut]
            test_indexes = indexes[cut:]
            
            yield [self.data[b:e+1,1:] for b,e in self.videos[train_indexes]], [self.data[b:e+1,1:] for b,e in self.videos[test_indexes]]
        

    def cross_validation(self, k=5):
        #k = 5 if k > 10 or k<1 else k
        fold_size = len(self.videos)//k
        indexes = list(range(len(self.videos)))
        random.shuffle(indexes)
        for fold in range(k):
            begin = fold*fold_size
            end = begin+fold_size
            if fold == k-1: end = len(self.videos)
            val_indexes = indexes[begin:end]
            train_indexes = [index for index in indexes if index not in val_indexes]
            yield [self.data[b:e+1,1:] for b,e in self.videos[train_indexes]], [(self.data[b:e+1,1:],(b,e)) for b,e in self.videos[val_indexes]]











if __name__ == "__main__":
    ActicipateDataset()
    



