from collections import deque
import numpy as np
import cv2
import csv
import os
import argparse
import math
from glob import glob
import pickle
import random
from generate_dynamic_star import *
import skimage as sk
from skimage import io, transform
from torchvision import transforms, models
from tqdm import tqdm
import threading
import torch

class DStarRGBHandSptDataset: 
    def __init__(self, dataset = "train", max_size = 24, alpha = 0.7, window = 3, sequence = None, transform = None, ):
        self.max_size = max_size
        self.channels=3
        self.transform = transform
        self.window = window
        self.alpha = alpha
        self.seq = sequence
        samples = glob("/notebooks/datasets/Montalbano/{}/sample_images/*".format(dataset))
        self.indexes = []
        total = 0
        # ones = 0
        for sample in samples:
            frames = glob("{}/*.png".format(sample))
            size = len(frames)//2
            frames.close()
            if self.seq is None:
                b = 0
                e = size
                self.indexes.append((sample,b,e))
            else:
                for i in range(size//self.seq):
                    b = i*self.seq
                    e = b + self.seq
                    self.indexes.append((sample,b,e))
            total += size
        # print("umbalancing {}".format(ones/total))
        print("{} -> Samples {} |  Sequences {} | Total images {} |".format(dataset,len(samples),len(self.indexes),total))
        

    def __len__(self):
        return len(self.indexes)


    def __getitem__(self,index):
        starRGB = DynamicStarRGB(self.window, self.channels, self.max_size, self.alpha)
        file,b,e = self.indexes[index]
        images,hands, label = starRGB.get_chunk(file,b,e)
        sample = {"images":images, "hands":hands,"label":label}
        if self.transform is not None:
            return self.transform(sample)
        return sample

    def __iter__(self):
        for index in range(len(self)):
            starRGB = DynamicStarRGB(self.window, self.channels, None, self.alpha)
            images, hands, label = starRGB.get_images(self.files[index])
            sample = {"images":images, "hands":hands,"label":torch.tensor([label]).long()}
            if self.transform is not None:
                yield self.transform(sample)
            yield sample


class DataTrasformation(object):
    def __init__(self, output_size, data_aug = True):
        self.output_size = output_size
        self.data_aug = data_aug
        
    def random_jitter(self, image):
        if np.random.rand(1) > 0.3:
            image = transforms.ColorJitter(*(np.random.rand(4)*0.3) )(image)
        return image

    def randon_crop(self, images):
        image = images[0]
        height, width = self.output_size
        y = 0 if image.shape[1] - height <= 0 else np.random.randint(0, image.shape[1] - height)
        x = 0 if image.shape[2] - width <= 0 else np.random.randint(0, image.shape[2] - width)
        assert image.shape[2] >= width
        assert image.shape[1] >= height
        images[0] = image[:,y:y+height,x:x+width,:]
        return images

    def crop_center(self, images, out):
        image = images[0]
        y,x = image.shape[1], image.shape[2]
        cropy, cropx = out
        startx = x//2-(cropx//2)
        starty = y//2-(cropy//2) 
        startx = 0 if startx + cropx >=x  else startx
        starty = 0 if starty + cropy >=y else starty
        images[0] = image[:,starty:starty+cropy,startx:startx+cropx,:]
        return images

    def random_rotation(self,images):
          # pick a random degree of rotation between 25% on the left and 25% on the right
            if np.random.rand(1) < 0.5:
                image = images[0]
                hand_left = images[1][0]
                hand_right = images[1][1]
                random_degree = np.random.uniform(-5, 5)
                image = np.array([sk.transform.rotate(image[i], random_degree) for i in range(len(image))])
                hand_left = np.array([sk.transform.rotate(hand_left[i], random_degree) for i in range(len(hand_left))])
                hand_left = np.array([sk.transform.rotate(hand_right[i], random_degree) for i in range(len(hand_right))])
                images[0] = image
                images[1][0] = hand_left
                images[1][1] = hand_right
            return images


    def random_noise(self,images):
        
        if np.random.rand(1) < 0.3:
            image = images[0]
            hand_left = images[1][0]
            hand_right = images[1][1]
            images[0] = sk.util.random_noise(image)
            images[1][0] = sk.util.random_noise(hand_left)
            images[1][1] = sk.util.random_noise(hand_right)

        return images
           

    def random_horizontal_flip(self,images):
        if np.random.rand(1) < 0.5:
            image = images[0]
            hand_left = images[1][0]
            hand_right = images[1][1]
            images[0] = image[:,:, ::-1,:]
            images[1][0] = hand_right[:,:, ::-1,:]
            images[1][1] = hand_left[:,:, ::-1,:]
        return images
           
   
    def __call__(self, sample):
        image = sample["images"]
        hands = sample["hands"]
        hands = [hands[:,:,:hands.shape[1]], hands[:,:,hands.shape[1]:]] #separet the right and left hands
        images = [image, hands]
        if self.data_aug:
            images = self.randon_crop(images)
            images = self.random_horizontal_flip(images)
            images = self.random_rotation(images)
            images = self.random_noise(images)


        else:
            # image = sk.transform.resize(image,self.output_size)
            images = self.crop_center(images, self.output_size)
        
        image = images[0]
        image = np.transpose(image, (0,3,1,2)).astype(np.float32)
        image = torch.from_numpy(image)/255.0
        sample["images"] = image

        hands = np.concatenate(images[1],2)
        hands = np.transpose(hands, (0,3,1,2)).astype(np.float32)
        hands = torch.from_numpy(hands)/255.0
        sample["hands"] = hands

        return sample


if __name__ == "__main__":
    #params = torch.load("dynamic_star_rgb_9527.pth")["params"]
    #print(params)
    star = DynamicStarRGB(window = 5, channels=3, max_size = None, alpha = 0.6)
    directories = ["train", "test", "validation"]
    total = [0, 0, 0]

    for i,dir in enumerate(directories):
        samples = glob.glob("/notebooks/datasets/Montalbano/{}/samples/*_color.mp4".format(dir))
        # print("/notebooks/datasets/Montalbano/{}/samples/*_color.mp4".format(dir))
        for sample in samples:
            name = sample.split("/")[-1].split("_")[0]
            images, labels = star.get_complete_images(sample)
            name = "/notebooks/datasets/Montalbano/{}/numpy_files/star_rgb_{}_{}".format( dir, name,len(labels))
            np.savez(name, images=images, labels=labels)
            total[i] += len(labels)
            print("{}  - {}/{}".format(name,len(images), len(labels)))
            
    print(total)
    for dir, t in zip(directories,total):
        print("Total {} = {}".format(dir,t))


