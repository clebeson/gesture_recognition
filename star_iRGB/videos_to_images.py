import scipy.misc as m
from glob import glob
import numpy as np
import pims
import os


files = glob("/home/clebeson/experiments/datasets/Montalbano/*/samples/*_color.mp4")
print(len(files))
for file in files:
    dir = file.replace("samples","sample_images").replace("_color.mp4","")
    os.makedirs(dir, exist_ok=True)
    print(dir)
    frames = pims.Video(file)
    frames = map(lambda f,s: m.imresize(f, tuple(s)).astype(np.uint8), frames, [[120,160]]*len(frames))
    for idx, frame in enumerate(frames):
        name = os.path.join(dir,"image_{}.png".format(idx))
        m.imsave(name,frame)
        
