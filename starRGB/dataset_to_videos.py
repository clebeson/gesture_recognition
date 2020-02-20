from __future__ import print_function
import numpy as np 
import scipy.ndimage as ndi 
import os 
import csv
import cv2
import shutil
import pims
import pickle
from moviepy.editor import *
import matplotlib.pyplot as plt
import scipy.misc as m
from itertools import izip as zip
import glob
from scipy.stats import norm
directory = '/home/clebeson/Documents/all/'
count = 0

def get_frames(video, begin = None, end = None, size = (240,320)):
    frames = []
    if len(video) < 1:
        return frames

    elif video[0].shape[:2] == size: 
        if begin is None or end is None: 
            begin = 0
            end = len(video)
            
        for i in range( len(video)):
            frames.append(video[i])
       
        for i in range( begin,end):
            frames.append(video[i])
    else:
        for i in range( begin,end):
            frame = m.imresize(video[i], size)
            frames.append(frame)

    return frames
def normalize(image):
    
    for i in range(3):
        img = image[:,:,i]
        max = np.max(img[:])
        min = np.min(img[:])
        image[:,:,i] = 255* (img-min) / (max-min) 
    return np.uint8(image)

def diff(img1, img2):
         return np.absolute(img1-img2)

def add(img1, img2):
    return img1 + img2

def get_star(frames, size = (128,96) ):
    #resize and convert to float32
    frames = list(map(lambda f,s: m.imresize(f, tuple(s)).astype(np.float32), frames, [list(size)]*len(frames)))
    return reduce(add, map(diff, frames[::2],frames[1::2]))





def star_grb(frames, size = (128,96), aug = None):
    total = len(frames)
    
    if aug is not None:
        t = total // 3
        mean = [t, t*2] 
        sigma = 0
        pos1 = np.zeros((1,int(aug)))
        pos2 = np.zeros((1,int(aug)))

        while (pos2-pos1 < 5).any():
            pos1 = (sigma  * norm.rvs(size=int(aug)) + t).astype(int)
            pos2 = (sigma  * norm.rvs(size=int(aug)) + t*2).astype(int)
    else:
        pos1, pos2 = [0],[(total//3)*2]
        
    images = []
    for p1,p2 in zip(pos1,pos2):
        star_R, star_G, star_B = np.zeros((size[0],size[1],3)), np.zeros((size[0],size[1],3)), np.zeros((size[0],size[1],3))
        
        img_before =  m.imresize(frames[0], size).astype(np.float32)
    # img_before =  m.imresize(frames[0], size).astype(np.uint8)
    # img_before =  cv2.cvtColor(img_before, cv2.COLOR_BGR2RGB ).astype(np.float32)
        
        for i in range(1,total):
            img_atual = m.imresize(frames[i], size).astype(np.float32)
            # img_atual = m.imresize(frames[i], size).astype(np.uint8)
            # img_atual = cv2.cvtColor(img_atual, cv2.COLOR_BGR2RGB ).astype(np.float32)
            image =  np.absolute( img_atual - img_before)
            img_before = img_atual 
            
            R, G, B= image[:,:,0], image[:,:,1], image[:,:,2]
            if i < p1:
                star_R[:,:,0] +=  R
                star_G[:,:,0] +=  G
                star_B[:,:,0] +=  B
            elif i < p2:
                star_R[:,:,1] +=  R
                star_G[:,:,1] +=  G
                star_B[:,:,1] +=  B
            
            else:
                star_R[:,:,2] +=  R
                star_G[:,:,2] +=  G
                star_B[:,:,2] +=  B
        


        # cv2.imshow('R', star_R.astype(np.uint8))
        # cv2.imshow('G', star_G.astype(np.uint8))
        # cv2.imshow('B', star_G.astype(np.uint8))
        # cv2.imshow('Bef',img_before.astype(np.uint8))
            


        # cv2.waitKey(2000)

        images.append(star_R)
        images.append(star_G)
        images.append(star_B)


    
    return map(normalize,images)


def generate_from_database():
    for root, subdirs, files in os.walk(directory):
        print(subdirs)
        if  len(files) == 0: continue
        roots = [root]*len(files)
        
        infos = []
        labels = np.zeros(( len(files)))
        for dir, file in zip(roots, files):
            
            path = os.path.join(dir, file)
            if path.endswith("color.mp4"):
                video = pims.Video(path)
                arch = file
            
            if path.endswith("labels.csv"):
                csvfile =  open(path, 'rb')
                infos = csv.reader(csvfile, delimiter=',', quotechar='|')

        size = len(video)
        rest_begin =  0
        for info in infos:
            label, begin, end = int(info[0]), int(info[1]), int(info[2])
            labels[begin:end]=label
            if end >= size: end = size-1
            
            name = directory + "result/"+ arch.replace(".mp4","")+ ("_{}_{}_{}.mp4".format(begin, end, label))
            frames = get_frames(video, begin, end)


            
            if frames == []: continue
            # new_clip = ImageSequenceClip(frames, fps=20.0)
            # new_clip.write_videofile(name) 
            
            star = star_grb(frames)
            if star == []: continue
            star_name = directory + "star_RGB/"+ arch.replace(".mp4","") + ("_{}_{}_{}.mp4".format(begin, end, label))
            star_clip = ImageSequenceClip(star, fps=1.5)
            star_clip.write_videofile(star_name) 
            

            if (begin > (rest_begin + 40)) and (begin < (rest_begin + 90)):
                rest_name = directory + "rest/result/"+ arch.replace(".mp4","")+ ("_{}_{}_{}.mp4".format(rest_begin, begin-1, 21))
                rest_frames = get_frames(video,rest_begin,begin)
                # rest_clip = ImageSequenceClip(rest_frames, fps=20.0)
                # rest_clip.write_videofile(rest_name) 

                star = star_grb(rest_frames)
                if star == []: continue
                star_name = directory + "rest/star_RGB/"+ arch.replace(".mp4","")+ ("_{}_{}_{}.mp4".format(rest_begin, begin-1, 21))
                star_clip = ImageSequenceClip(star, fps=1.5)
                star_clip.write_videofile(star_name) 
            
            rest_begin = end


def labeling_videos():
    labels_dict = {}
    for root, subdirs, files in os.walk(directory):
        
        if  len(files) == 0: continue
        roots = [root]*len(files)
        
        infos = []
        for dir, file in zip(roots, files):
            
            path = os.path.join(dir, file)
            if path.endswith("color.mp4"):
                video_path = path
                video = pims.Video(video_path)
                arch = file
            
            if path.endswith("labels.csv"):
                csvfile =  open(path, 'rb')
                infos = csv.reader(csvfile, delimiter=',', quotechar='|')

        size = len(video)
        labels = np.zeros((size))

        
        rest_begin =  0
        for info in infos:
            label, begin, end = int(info[0]), int(info[1]), int(info[2])
            if end >= size: end = size-1
            labels[begin:end]=label
       
        print(arch)
        dest =  directory.replace("all", "complete") + arch
        shutil.copy2(video_path, dest)
        labels_dict[arch] = labels
        # print(dest)
        # star_clip = ImageSequenceClip(star, fps=1.5)
        # star_clip.write_videofile(star_name) 
    with open("/home/clebeson/Documents/complete/labels.pkl","wb") as filehandler:
        pickle.dump(labels_dict,filehandler)






def generate_from_videos():
    dir = directory+"*.mp4"
    files =  glob.glob(dir)


    for file in files :
        name = (file.split("/")[-1])
        path = file.replace("RGB/"+name,"")
        name = name.replace("color","color_{}")

        video = pims.Video(file)
        rest_begin =  0

        frames = get_frames(video)
        size = (96,128)
        n = 1
        star = star_grb(frames,size, n)
        if star == []: continue
        for i in range(0,n):
            star_name = "{}star_rgb_{}-{}_aug/{}".format(path, size[1],size[0], name.format(i))
            star_clip = ImageSequenceClip(star[i*3:i*3+3], fps=1.5)
            star_clip.write_videofile(star_name) 
        
    

# generate_from_database()

labeling_videos()
# with open("/home/clebeson/Documents/complete/labels.pkl",'rb') as file:
#     object_file = pickle.load(file)
#     for key in object_file:
#         label = object_file[key]
#         plt.bar(np.arange(len(label)),label)

#         plt.ylabel('Number of Samples')
#         plt.xlabel('Classes')
#         plt.show()

#         print(key)





                
                
                

