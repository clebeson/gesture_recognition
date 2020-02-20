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
import glob
from scipy.stats import norm
import matplotlib.pyplot as plt
import pickle
from numpy.linalg import norm
from  scipy.spatial.distance import cosine

#Nomalize eache frame based on a genet=rated star
#Used for generating a video
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
    return img

def add(img1, img2):
    img = img1 + img2
    return img


def euclidian_distance(img1,img2):
    size = img1.shape
    norm1 = norm(img1,axis=2)
    norm2 = norm(img2,axis=2)
    dot = np.sum(np.multiply(img1,img2),axis = 2)
    cos = dot/(norm1*norm2 + 1e-18  )
    lamb = 1-cos
    # lamb[np.where(cos < 0.2) ] = 0.0
    d0 = np.absolute(norm1-norm2)
    return (1 - lamb/2.0)*d0





def get_star(frames, func = "cos" ):
    if func == "cos":
        return (reduce(add, map(euclidian_distance, frames[:-1],frames[1:])) )
    return (reduce(add, map(diff, frames[:-1],frames[1:])) )

    

# def get_star_1_RGB1(frames):
#     total = len(frames) 
    
#     # if list(frames[0].shape)[-1] == 3:
#     #     frames = list(map(cv2.cvtColor, frames, [cv2.COLOR_RGB2GRAY]*total))
#     step = total //3
#     r = np.expand_dims(get_star1(frames[:step]), 2)
#     g = np.expand_dims(get_star1(frames[step:step*2]), 2)
#     b = np.expand_dims(get_star1(frames[step*2:]), 2)
    
    # return (np.concatenate([r,g,b], axis = 2))

def get_star_1_RGB(frames):
    total = len(frames) 
    
    # if list(frames[0].shape)[-1] == 3:
    #     frames = list(map(cv2.cvtColor, frames, [cv2.COLOR_RGB2GRAY]*total))
    step = total //3
    r = np.expand_dims(get_star(frames[:step]), 2)
    g = np.expand_dims(get_star(frames[step:step*2]), 2)
    b = np.expand_dims(get_star(frames[step*2:]), 2)
    
    return (np.concatenate([r,g,b], axis = 2))

def get_star_3_RGB(frames):
    total = len(frames)
    step = total //3
    frames_1 = list(map(lambda f: f[:,:,0], frames))
    frames_2 = list(map(lambda f: f[:,:,1], frames))
    frames_3 = list(map(lambda f: f[:,:,2], frames))
    image1 = get_star_1_RGB(frames_1)
    image2 = get_star_1_RGB(frames_2)
    image3 = get_star_1_RGB(frames_3)
    return np.concatenate([image1,image2,image3], axis = 2)
    

def get_frames(file, size = (120,160)):
    frames = pims.Video(file)
    frames = list(map(lambda f,s: m.imresize(f, tuple(s)).astype(np.float32), frames, [list(size)]*len(frames)))
    return frames

def generate_star(video):
    if isinstance(video, basestring):
        frames = get_frames(video)
    else:
        frames = video
    # if list(frames[0].shape)[-1] == 3:
    #     frames = list(map(cv2.cvtColor, frames, [cv2.COLOR_RGB2GRAY]*len(frames)))
    
    star1 = get_star_1_RGB(frames)

    frames = list(map(cv2.cvtColor, frames, [cv2.COLOR_RGB2GRAY]*len(frames)))
    star = get_star_1_RGB(frames)
    #print(star-star1)
    return [normalize(star), normalize(star1),normalize(star1-star)]

def get_label(name):
    n, _ = os.path.splitext(name)
    n = n.split("/")[-1]
    n = n.split("_")[-1]
    return int(n) -1 




# images.extend(star)
# print(3)
# videos = map(pims.Video, files[200:])
# star = list(map(normalize, map(get_star,videos)))
# images.extend(star)

# images = []
# # total = len(files)
# # simze = len(files) // 500
# for i,file in enumerate(files): 
#     if i%100 == 0:
#         print(i)
#     frames = get_frames(file)
#     # star = normalize(get_star(frames) )
#     star = get_star_1_RGB(frames)
    
#     # cv2.imshow("1", star[1])
#     # cv2.imshow("2", star[2])
#     cv2.waitKey()
#     images.append(star)
#     # m.imsave(file.replace("mp4","png").replace("basta", "result"), star)
    





N = 30 #window in the image sequence
alpha = 1.0 #forget term alpha
inv_alpha = 1.0/alpha #alpha**-1
num_stars = 3 #number of stars (RGB = 3)
buffer = [np.zeros((120,160))]*N #buffering the last N distances betwee two images. Initialized as zero
indices = [i*N//num_stars for i in range(num_stars)]+[N]
frames = get_frames("/home/clebeson/Downloads/trainning4/Sample00301/Sample00301_color.mp4", (120,160))
created_star = False
stars = np.zeros((120,160,num_stars)) #initialize an star

# before = np.copy(stars)
# data = {"mean":[], "var":[]}
for k in range(1,len(frames)):
    dist_cos = euclidian_distance(frames[k-1],frames[k])
    buffer.append(dist_cos)
    del buffer[0] 
    elif not created_star:
        images = [img for img in buffer]
        for i in range(num_stars):
            stars[:,:,i] = reduce(add, images[indices[i]:indices[i+1]]) / (indices[i+1] - indices[i])
        created_star = True
    
    else: 
        for i in range(num_stars):
            n = (indices[i+1] - indices[i])
            stars[:,:,i] = stars[:,:,i]*alpha + (buffer[indices[i+1]] - np.power(alpha,n) * buffer[indices[i]]) / n
    
        # print(np.mean((stars-before)))
        # diff_img = np.absolute(np.power(stars-before,2))
        # diff_img = diff_img[np.where(diff_img > 5)]
        # m = np.mean(diff_img)
        # v = np.var(diff_img)
        # data["mean"].append(m if not np.isnan(m) else 0.0)
        # data["var"].append(v if not np.isnan(v) else 0.0)
        # before = np.copy(stars)
        # buffer.popleft()
        

    for i in range(num_stars):
        #cv2.imshow("normalized_{}".format(i), np.uint8(stars[:,:,i]))
        cv2.imshow("normalized_{}".format(i), 127 * np.uint8(stars))
    cv2.waitKey()
    



import matplotlib.pyplot as plt
import numpy as np
import wave
import sys

def normi(data, inf = 0, sup=1):
    array= np.array(data).astype(float)
    min = array.min()
    max = array.max()
    print(min,max)
  
    array = (array - float(min)) 
    array = array / float(array.max())
    print(array.min(),array.max())

    return array
    

spf = wave.open('/home/clebeson/Downloads/trainning4/Sample00301/Sample00301_audio.wav','r')

#Extract Raw Audio from Wav File
signal = spf.readframes(-1)
signal = np.fromstring(signal, 'Int16')
sample = len(signal)//1200
indices = np.arange(1200)*sample

#If Stereo
if spf.getnchannels() == 2:
    print('Just mono files')
    sys.exit(0)

plt.figure(1)
plt.title('voice')
plt.plot(normi(signal[indices] ))
# plt.show()

plt.figure(2)
plt.title('mean')
plt.plot(normi(data["mean"][5:]) )

plt.figure(1)
plt.title('var')
plt.plot(normi(data["var"][5:]))
plt.show()





# ind = range(50,93) + range(94,127) + range(130,171) + range(174,207) + range(230,273) + range(310,353) + range(370,416) + range(431,479) + range(491,534) + range(535,572) + range(591,630) + range(650,690) + range(712,761) + range(792,827) + range(838,881) + range(882,906) + range(951,991) + range(996,1028) + range(1053,1106) + range(1112,1159) 
# labels = np.zeros((1159))+2
# labels[ind] = 0
# plt.figure(2)
# plt.title('var')
# plt.plot(labels)
# plt.plot(data["var"])
# plt.plot(data["mean"])
# plt.show()
