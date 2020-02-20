import numpy as np 
import moviepy as moviepy
import os
import glob
from moviepy.editor import *


root = "/home/clebeson/Downloads/GRIT_DATASET_5584ADHUHA_2124fGFZZ/HRI_GestureDataset_Tsironi/"
labels = ["abort", "circle", "hello", "no", "stop", "turn", "turn_left", "turn_right", "warn"]
def compare(im1,im2):
    im1 = "".join(im1.split(":")[-2:])
    im2 = "".join(im2.split(":")[-2:])
    if im1 > im2: return 1
    if im1 == im2: return 0
    return -1


dirs = os.listdir(root)
for dir in dirs:
    subdirs = os.listdir(os.path.join(root,dir))
    for subdir in subdirs:
        images = glob.glob(os.path.join(root,dir,subdir,"*.jpg"))
        images.sort(cmp=compare)
       
        label= labels.index(dir)
        name = "/home/clebeson/Downloads/GRIT_DATASET_5584ADHUHA_2124fGFZZ/videos/grit-{}-{}.mp4".format(subdir ,label)
        
        new_clip = ImageSequenceClip(images, fps=10.0)
        new_clip.write_videofile(name)        