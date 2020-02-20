from __future__ import print_function
import numpy as np 
import scipy.ndimage as ndi 
import os 
import csv
import cv2
import pims
from functools import reduce
from moviepy.editor import *
import scipy.misc as m
import glob
import time
from scipy.stats import norm
import matplotlib.pyplot as plt
import pickle



star_image = []
def normalize2(image, star):
    if len(list(image.shape)) == 3:
        for i in range(3):
            img = star[:,:,i]
            max = np.max(img[:])
            min = np.min(img[:])
            image[:,:,i] = 255* (img-min) / (max-min) 
    else:
        max = np.max(star[:])
        min = np.min(star[:])
        image = 255* (image-min) / (max-min) 
    return np.uint8(image)
def normalize(image):
    if len(list(image.shape)) == 3:
        for i in range(3):
            img = image[:,:,i]
            max = np.max(img[:])
            min = np.min(img[:])
            image[:,:,i] = 255* (img-min) / (max-min) 
    else:
        max = np.max(image[:])
        min = np.min(image[:])
        image = 255* (image-min) / (max-min) 

    return np.uint8(image)


def diff(img1, img2):
    img = np.absolute(img1-img2)
    star_image.append(img)
    return img

def add(img1, img2):
    img = img1 + img2
    return img

def get_star(frames ):
    #resize and convert to float32
    return np.sum(np.absolute(frames[:-1]-frames[1:]),0)
    # return (reduce(add, map(diff, frames[:-1],frames[1:])) )

def get_star_1_RGB(frames):
    
    total = len(frames) 
    if list(frames[0].shape)[-1] == 3:
        frames = np.array(list(map(cv2.cvtColor, frames, [cv2.COLOR_RGB2GRAY]*total)))
    step = total //3
    r = np.expand_dims(get_star(frames[:step]), 2)
    g = np.expand_dims(get_star(frames[step:step*2]), 2)
    b = np.expand_dims(get_star(frames[step*2:]), 2)
    del frames
    return normalize(np.concatenate([r,g,b], axis = 2))

def get_star_3_RGB(frames):
    total = len(frames)
    step = total //3

    frames_1 = list(map(lambda f: f[:,:,0], frames))
    frames_2 = list(map(lambda f: f[:,:,1], frames))
    frames_3 = list(map(lambda f: f[:,:,2], frames))
    image1 = get_star_1_RGB(frames_1)
    image2 = get_star_1_RGB(frames_1)
    image3 = get_star_1_RGB(frames_1)
    return np.concatenate([image1,image2,image3], axis = 2)
    

def get_frames(file, size = (120,160)):
    frames = pims.Video(file)
    frames = list(map(lambda f,s: m.imresize(f, tuple(s)).astype(np.float32), frames, [list(size)]*len(frames)))
    return np.array(frames)

#files = glob.glob("/home/clebeson/basta/*.mp4")
files = ["/home/clebeson/Dropbox/Sample0004_color_163_200_13.mp4"]
# print(2)

def generate_star(video):
    
    if isinstance(video, basestring):
        frames = get_frames(video)
    else:
        frames = video

    if list(frames[0].shape)[-1] == 3:
        frames = np.array(list(map(cv2.cvtColor, frames, [cv2.COLOR_RGB2GRAY]*len(frames))))
 
    star = normalize(get_star(frames))

    cv2.imshow("2", star)
    cv2.waitKey()




    





# images.extend(star)
# print(3)
# videos = map(pims.Video, files[200:])
# star = list(map(normalize, map(get_star,videos)))
# images.extend(star)

# images = []
# # total = len(files)
# # simze = len(files) // 500
# for i,file in enumerate(files[:1]): 
#     if i%100 == 0:
#         print(i)
#     frames = get_frames(file)
#     star = normalize(get_star(frames) )
#     star = get_star_1_RGB(frames)
    
#     # cv2.imshow("1", star[1])
#     # cv2.imshow("2", star[2])
#     cv2.waitKey()
#     images.append(star)
#     # m.imsave(file.replace("mp4","png").replace("basta", "result"), star)
    

# images = []
# for star, frame in zip(video, frames):
#     image = np.ones((star.shape[0], 2*star.shape[1] + 10, star.shape[2]) )
#     print(frame.shape)
#     image[:,0:star.shape[1],:] = normalize(star, star1)
#     image[:,star.shape[1]+10:,:] = frame
#     images.append(image)

# star_clip = ImageSequenceClip(images, fps=3.0)
# star_clip.write_videofile("video_star.mp4") 

# videos = map(pims.Video, files)
# images = list(map(normalize, map(get_star,videos)))

# with open("/home/clebeson/bata.pkl", 'wb') as f:
#     pickle.dump(images, f)


# frames= get_frames("/home/clebeson/Dropbox/Sample0004_color_163_200_13.mp4", (240,320))
# frames = list(map(cv2.cvtColor, frames, [cv2.COLOR_RGB2BGR]*len(frames)))
# #star = get_star(frames)
# #frames = np.transpose(np.array(star_image), (2,0,1))
# #star = get_star(frames)
# for i, image in enumerate(frames):
#     #m.imsave("/home/clebeson/Pictures/basta{}.png".format(i), normalize2(image, star))
#     cv2.imshow("2", np.uint8(image) )
#     cv2.waitKey()

# video = "/home/clebeson/Dropbox/Sample0004_color_163_200_13.mp4"
frames= get_frames("Sample0437-530-559-video_13.mp4", (240,320))
frames = np.array(frames)
print(frames.shape)
frames = np.rot90(frames)
print(frames.shape)
# frames = list(map(cv2.cvtColor, frames, [cv2.COLOR_RGB2GRAY]*len(frames)))
# frames = list(map(diff, frames[:-1],frames[1:]))

# rgb3 = get_star_3_RGB(frames)
# start = time.time()
# for _ in  range(100):
#     rgb1 = get_star_1_RGB(frames)
# print((time.time()-start))
# print(rgb1.max())
# m.imsave("/home/clebeson/Documents/basta/rgb_result.png", rgb1)
# m.imsave("/home/clebeson/Documents/basta/r_result.png", rgb1[:,:,0])
# m.imsave("/home/clebeson/Documents/basta/g_result.png", rgb1[:,:,1])
# m.imsave("/home/clebeson/Documents/basta/b_result.png", rgb1[:,:,2])
# m.imsave("/home/clebeson/Documents/basta/rgb_R.png", rgb3[:,:,:3])
# m.imsave("/home/clebeson/Documents/basta/rgb_G.png", rgb3[:,:,3:6])
# m.imsave("/home/clebeson/Documents/basta/rgb_B.png", rgb3[:,:,6:9])
# for i, image in enumerate(frames):
#     m.imsave("/home/clebeson/Documents/basta/{}.png".format(i), normalize(image))
    # cv2.imshow("2", np.uint8(image))
    # cv2.waitKey()
# generate_star(video)

#Sample0437-530-559-video_13.mp4"