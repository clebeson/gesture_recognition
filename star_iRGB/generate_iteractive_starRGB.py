from __future__ import print_function
import numpy as np 
import scipy.ndimage as ndi 
import os 
import csv
import cv2
import pims
import time
from functools import reduce
from moviepy.editor import *
import scipy.misc as m
from glob import glob
import imageio
from scipy.stats import norm
import matplotlib.pyplot as plt
import pickle
from numpy.linalg import norm
from  scipy.spatial.distance import cosine
import pandas as pd
from scipy.ndimage import gaussian_filter as gaussian

class DynamicStarRGB:
    def __init__(self, window = 3, channels = 3, max_size = 64, alpha = 0.6, spotting = False):
        self.window = window
        self.max_size = max_size
        self.alpha = alpha #forget term alpha
        self.inv_alpha = 1.0/alpha #alpha**-1
        self.channels = channels
        self.spotting = spotting
        self.buffer = [np.zeros((120,160)) for _ in range(window*channels)] #buffering the last N distances betwee two images. Initialized as zero
        self.image = np.zeros((120,160,channels))
        self.frames = []


    def normalize(self, image):
        max = image.max()
        min = image.min()
        image =  (image-min) / (max-min) 
        return image

    def get_images_only(self, file):
        if not self.frames: self.set_video(file, hands = False)
        max_size = self.max_size if self.max_size is not None else len(self.frames)
        beta = 1-self.alpha
        beta_N = np.power(beta, self.window)

        self.stars = np.zeros((max_size,120,160,self.channels)).astype(np.uint8)
        for index in range(1, max_size):
            if index < len(self.frames):
                dist_cos = self.cosine_distance(self.frames[index-1],self.frames[index])
                self.buffer.append(dist_cos)
                for i in range(self.channels):
                    channel = self.channels-i
                    e = self.window*channel
                    b = e - self.window
                    self.image[:,:,i] = self.image[:,:,i]*beta + self.alpha*(self.buffer[e] - beta_N * self.buffer[b])
                del self.buffer[0] 

            self.stars[index] = np.uint8(255*self.normalize(self.image))
                      
        return self.stars, self.get_label(file)


    def get_complete(self, file):
        if not self.frames: self.set_video(file, hands = False)
        max_size = 10
        beta = 1-self.alpha
        beta_N = np.power(beta, self.window)
        self.stars = np.zeros((max_size,120,160,self.channels)).astype(np.uint8)
        sep_channels = np.zeros((max_size,120,160,self.channels)).astype(np.uint8)

        for index in range(1, max_size):
            if index < len(self.frames):
                dist_cos = self.cosine_distance(self.frames[index-1],self.frames[index])
                self.buffer.append(dist_cos)
                for i in range(self.channels):
                    channel = self.channels-i
                    e = self.window*channel
                    b = e - self.window
                    self.image[:,:,i] = self.image[:,:,i]*beta + self.alpha*(self.buffer[e] - beta_N * self.buffer[b])
                    sep_channels[index,:,:,i] = 255*self.normalize(self.image[:,:,i]*beta + self.alpha*(self.buffer[e] - beta_N * self.buffer[b]))
                del self.buffer[0] 
            self.stars[index] = np.uint8(255*self.normalize(self.image))
            
        return self.stars,  sep_channels, np.array([self.normalize(b) for b in self.buffer])



    def get_images(self, file):
        if not self.frames: self.set_video(file, hands = True, factor=1)
        max_size = self.max_size if self.max_size is not None else len(self.frames)
        beta = 1-self.alpha
        beta_N = np.power(beta, self.window)
        self.stars = np.zeros((max_size,120,160,self.channels)).astype(np.uint8)
        hands = np.zeros((max_size, 40,80,3)).astype(np.uint8)
        hand =  np.zeros((40,80,3)).astype(np.uint8)
        for index in range(1, max_size):
            if index < len(self.frames):
                dist_cos = self.cosine_distance(self.frames[index-1],self.frames[index])
                self.buffer.append(dist_cos)
                for i in range(self.channels):
                    channel = self.channels-i
                    e = self.window*channel
                    b = e - self.window
                    self.image[:,:,i] = self.image[:,:,i]*beta + self.alpha*(self.buffer[e] - beta_N * self.buffer[b])
                del self.buffer[0] 
            
            if index < len(self.hands): 
                hand = self.hands[index]
            
            hands[index] = hand
            self.stars[index] = np.uint8(255*self.normalize(self.image))
            
        return self.stars,  hands, 1 if self.spotting else self.get_label(file)

    def get_chunk(self, file, begin = None, end = None):
        frames = glob("{}/*.png".format(file))
        frames.sort(key=lambda x: int(x.replace(".png","").split("_")[-1]))
        begin = 0 if begin is None else begin
        end = len(frames)//2 if end is None else end
        self.frames = [imageio.imread(frame).astype(np.float32) for frame in frames[begin*2:end*2:2]]
        hands = np.load(file.replace("sample_images","samples")+"_hands.npy")
        self.hands = np.uint8(hands[begin*2:end*2:2])
        self.hands = list(map(lambda f,s: m.imresize(f, tuple(s)).astype(np.float32), self.hands, [[40,80]]*len(self.hands)))
        images, hands , _ = self.get_images(file)
        labels = self.get_complete_label(file, len(frames))
        labels = labels[begin:end]
        return images, hands, np.where(labels>0,1,0)

    def get_chunk_images(self, file, begin = None, end = None):
        frames = glob("{}/*.png".format(file))
        frames.sort(key=lambda x: int(x.replace(".png","").split("_")[-1]))
        begin = 0 if begin is None else begin
        end = len(frames) if end is None else end
        self.frames = [imageio.imread(frame).astype(np.float32) for frame in frames[begin:end]]
        hands = np.load(file.replace("sample_images","samples")+"_hands.npy")
        self.hands = np.uint8(hands[begin:end])
        self.hands = list(map(lambda f,s: m.imresize(f, tuple(s)).astype(np.float32), self.hands, [[40,80]]*len(self.hands)))
        images, hands , _ = self.get_images(file)
        labels = self.get_complete_label(file, len(frames),factor = 1)
        labels = labels[begin:end]
        return images, hands, labels #np.where(labels>0,1,0)


    def get_StarRGB_and_images(self, file):
        if not self.frames: self.set_video(file, hands = False)
        max_size = self.max_size if self.max_size is not None else len(self.frames)
        beta = 1-self.alpha
        beta_N = np.power(beta, self.window)
        self.stars = np.zeros((max_size,120,160,self.channels), dtype = np.uint8 )
        context = np.zeros((max_size, 120,160,3), dtype = np.uint8 )
        for index in range(1, max_size):
            if index < len(self.frames):
                frame = self.frames[index]
                dist_cos = self.cosine_distance(self.frames[index-1],self.frames[index])
                self.buffer.append(dist_cos)
                for i in range(self.channels):
                    channel = self.channels-i
                    e = self.window*channel
                    b = e - self.window
                del self.buffer[0] 
            
            self.stars[index] = np.uint8(255*self.normalize(self.image))
            context[index] = frame
            
        return self.stars,  context, self.get_label(file)


    def cosine_distance(self,img1,img2,w=1.0):
        """Calculates the cosine distance between img1 and imag2 and multiplies the result by a weight w """
        norm1 = np.linalg.norm(img1,axis=2)
        norm2 = np.linalg.norm(img2,axis=2)
        dot = np.sum(np.multiply(img1,img2), axis = 2)
        cos = dot/(norm1*norm2 + 1e-8)
        lamb = 1-cos
        #eliminate the "ghosts" in the image represented by small angles
        lamb[lamb > 0.001] = 0.0 
        d0 = np.absolute(norm1-norm2)
        result = (1.0 - lamb/2.0) * d0 * w
        return result

    def euclidian_distance(self, img1,img2,w = 1.0):
        """Calculates the euclidian distance between img1 and imag2 and multiplies the result by a weight w"""
        return np.linalg.norm(img1-img2,axis=2)*w



    def set_video(self, file, hands = False, size = (120,160), factor = 2):
        video =  pims.Video(file)
        frames = video[::factor]
        self.frames = list(map(lambda f,s: m.imresize(f, tuple(s)).astype(np.float32), frames, [list(size)]*len(frames)))
        video.close()
        if hands:
            self.hands = np.uint8(np.load(file.replace(".mp4",".npy").replace("_color",""))[::factor])
        

    def get_label(self,name):
        n, _ = os.path.splitext(name)
        n = n.split("_")[-1]
        return int(n)-1 

    
    def get_complete_label(self,name,size, factor = 2):
        name = name.replace("sample_images","samples")+"_labels.csv"
        csv = pd.read_csv(name,names = ["label","begin","end"])
        labels = np.zeros(size).astype(int)
        for _,row in csv.iterrows():
            labels[int(row["begin"]):int(row["end"])] = int(row["label"])
        return labels[::factor]

         





if __name__ == "__main__":
    # frames = get_frames("Sample0036_depth.mp4", (120,160))
    total = 0
    times = 0
    for i in range(100):
        iter = DynamicStarRGB(3,3,100, 0.6)
        # start = time.time()
        images, l= iter.get_images_only("Sample0004_163_200_13.mp4")
        times += l
    print(times/10000)

        
        # # iter = DynamicStarRGB(5,3,65, 0.6)
        # # # iter.frames = frames
        # # images, labels = iter.get_comimages("Sample0036_color.mp4")
        # # print(images.shape)
        # for i in range(len(images)):
        #     m.imsave("basta/star_{}_R.png".format(i), images[i,:,:,0])
        #     m.imsave("basta/star_{}_G.png".format(i),images[i,:,:,1])
        #     m.imsave("basta/star_{}_B.png".format(i),images[i,:,:,2])
        #     m.imsave("basta/channel_{}_R.png".format(i), ch[i,:,:,0])
        #     m.imsave("basta/channel_{}_G.png".format(i),ch[i,:,:,1])
        #     m.imsave("basta/channel_{}_B.png".format(i),ch[i,:,:,2])
        #     m.imsave("basta/dist_{}.png".format(i),a[i])
            # image = np.concatenate([image[:,:,0],image[:,:,1],image[:,:,2]],1)
            # print(i, image.shape)
            # cv2.imshow("normalized", image)
            # cv2.waitKey(1000)
        
        


