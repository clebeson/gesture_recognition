from __future__ import print_function
import numpy as np 
import scipy.ndimage as ndi 
import os 
import csv
import pims
from moviepy.editor import *
import scipy.misc as m
import glob
import cv2
from scipy.stats import norm
import logging
# directory = '/home/clebeson/Documents/train_split_half/*.mp4' 
logging.getLogger("moviepy").setLevel(logging.WARNING)
# directory = name = "/home/clebeson/Downloads/GRIT_DATASET_5584ADHUHA_2124fGFZZ/videos/*.mp4"
directory = "/home/clebeson/Documents/Database/data1/train/RGB*/*.mp4"
samples = glob.glob(directory)

n = 10.0
print("Total:" ,len(samples)*n)
count_image = 0
for sample in samples:
    print("image: ",count_image)
    #vid =  '{}/{}/{}/{}_color.mp4'.format(directory,data,sample,sample)
    #labels_file = '{}/{}/{}/{}_labels.csv'.format(directory,data,sample,sample)
    #labels_file = '{}/{}/{}_labels.csv'.format(directory,"labels",sample)
    clip = np.array(pims.Video(sample))  #.cutout(0, 7)
    
    size = len(clip)
    t = size // 3
    mean = [t, t*2] 
    sigma = 4.0
    pos1 = np.zeros((1,int(n//2)))
    pos2 = np.zeros((1,int(n//2)))

    while (pos2-pos1 < 5).any():
        pos1 = (sigma  * norm.rvs(size=int(n//2)) + t).astype(int)
        pos2 = (sigma  * norm.rvs(size=int(n//2)) + t*2).astype(int)
       
    
    star_RR = np.zeros((240, 320, 1))
    star_RG = np.zeros((240, 320, 1))
    star_RB = np.zeros((240, 320, 1))

    star_GR = np.zeros((240, 320, 1))
    star_GG = np.zeros((240, 320, 1))
    star_GB = np.zeros((240, 320, 1))

    star_BR = np.zeros((240, 320, 1))
    star_BG = np.zeros((240, 320, 1))
    star_BB = np.zeros((240, 320, 1))

    img_before = clip[0].astype(float)
    images = []
    id = 0
    for p1,p2 in zip(pos1,pos2):
     
        for i in range(1,size):
            image_color =clip[i].astype(float)
            image = np.uint8(np.absolute( image_color - img_before) )
            img_before = image_color
            R, G, B= np.expand_dims(image[:,:,0],2), np.expand_dims(image[:,:,1],2), np.expand_dims(image[:,:,2],2)
        
            
            if i < p1:
                star_RR = star_RR + R
                star_GR = star_GR + G
                star_BR = star_BR + B

            elif i < p2:
                star_RG = star_RG + R
                star_GG = star_GG + G
                star_BG = star_BG + B
            
            else:
                star_RB = star_RB + R
                star_GB = star_GB + G
                star_BB = star_BB + B

        
            
        image_R = np.uint8(np.concatenate((star_RR,star_RG,star_RB), 2))
        image_G = np.uint8(np.concatenate((star_GR,star_GG,star_GB), 2))
        image_B = np.uint8(np.concatenate((star_BR,star_BG,star_BB), 2))
        image_R = m.imresize( image_R, 0.25 )
        image_G = m.imresize( image_G, 0.25 )
        image_B = m.imresize( image_B, 0.25 )
        
        # cv2.imshow("R", np.uint8(image_R)) 
        # cv2.imshow("G", np.uint8(image_G)) 
        # cv2.imshow("B", image_B) 
            
        # cv2.waitKey(3000)
        # print(image_B.shape)
        

        images.append(image_R)
        images.append(image_G)
        images.append(image_B)

        name = sample.replace("RGB", "aug_RGB_80").replace("_color_", "_color_{}_".format(id))
        # name = sample.replace("videos", "videos_star")

        # name = "/home/clebeson/Documents/test_split/{}-{:d}-{:d}-{}-video_{}.mp4".format(sample,start,id,end,lbl)
        new_clip = ImageSequenceClip(images, fps=1)
        new_clip.write_videofile(name) 
        del images[:]
                    
        id += 1
    del clip
    count_image += 1
    
      
        
                
                
                

