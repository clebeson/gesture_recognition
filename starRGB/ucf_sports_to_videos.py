from __future__ import print_function
import numpy as np 
import scipy.ndimage as ndi 
import os 
import csv
from moviepy.editor import *
import scipy.misc as m  
import glob
from PIL import Image
from scipy import ndimage as ndi

w,h, c= 100,100, 3

def rot_5_pos(img):
    return ndi.interpolation.rotate(img,5)

def rot_5_neg(img):
    return ndi.interpolation.rotate(img,-5)

def britness(img):
    img1 = img+img*0.2
    return ((img1-min(img1))*255 ) // max(img1-min(img1))

def noising(img):
    return (img + np.random.standard_normal((w,h,c)) )

def flip(img):
    return np.fliplr(img)

base_dir = "/home/clebeson/Downloads/ucf_sports_actions/ucf_action" 
dest_dir = "/home/clebeson/Documents/ucf_sports"  
images_geral =  [] 
for dir in os.listdir(base_dir):
    print(dir)
    for file in os.listdir(base_dir+"/"+dir):
        print("----> ",file)
        images = []
        if os.path.isdir(base_dir+"/"+dir+"/"+file):
           names =  sorted(glob.glob(base_dir+"/"+dir+"/"+file+"/*.jpg"))
           print("--------->",len(names))
           for name in names:
                im = m.imresize(m.imread(name), (w,h))
                images.append(im)

           imagesf = map(flip, images)

           images1 = map(rot_5_pos, images)
           images1f = map(flip, images1)

           images2 = map(rot_5_neg,  images)
           images2f = map(flip,  images2)

           images3 = map(noising, images)
           images3f = map(flip, images3)

           new_clip = ImageSequenceClip(images, fps = 20.0)
           video_name = dest_dir+"/"+dir+"_"+file+"_0.mp4"
           new_clip.write_videofile(video_name) 
           new_clip = ImageSequenceClip(imagesf, fps = 20.0)
           video_name = dest_dir+"/"+dir+"_"+file+"_0f.mp4"
           new_clip.write_videofile(video_name) 

           new_clip = ImageSequenceClip(images1, fps = 20.0)
           video_name = dest_dir+"/"+dir+"_"+file+"_1.mp4"
           new_clip.write_videofile(video_name)
           new_clip = ImageSequenceClip(images1f, fps = 20.0)
           video_name = dest_dir+"/"+dir+"_"+file+"_1f.mp4"
           new_clip.write_videofile(video_name)

           new_clip = ImageSequenceClip(images2, fps = 20.0)
           video_name = dest_dir+"/"+dir+"_"+file+"_2.mp4"
           new_clip.write_videofile(video_name)
           new_clip = ImageSequenceClip(images2f, fps = 20.0)
           video_name = dest_dir+"/"+dir+"_"+file+"_2f.mp4"
           new_clip.write_videofile(video_name)

           new_clip = ImageSequenceClip(images3, fps = 20.0)
           video_name = dest_dir+"/"+dir+"_"+file+"_3.mp4"
           new_clip.write_videofile(video_name)
           new_clip = ImageSequenceClip(images3f, fps = 20.0)
           video_name = dest_dir+"/"+dir+"_"+file+"_3f.mp4"
           new_clip.write_videofile(video_name)
