from __future__ import print_function
import numpy as np 
import scipy.ndimage as ndi 
import os 
import csv
import pims
from functools import reduce
from moviepy.editor import *
import scipy.misc as m
import glob
from scipy.stats import norm
import matplotlib.pyplot as plt
import pickle
from  scipy.spatial.distance import cosine
import imageio
imageio.plugins.ffmpeg.download()


def normalize(image):
    """Normalizes image between 0 and 1"""
    max = np.max(image[:])
    min = np.min(image[:])
    image = (image-min) / (max-min+1.0) 
    return image



def diff(img1, img2, w=1.0):
    """Calculate absolute difference between  img1 and imag2 and multiplies by a weight w """
    if img1 is None or img2 is None: return 0
    img1 = img1.mean(2, keepdims=False) if len(img1.shape) == 3 else img1
    img2 = img2.mean(2, keepdims=False) if len(img2.shape) == 3 else img2
    return np.absolute(img1-img2)*w

def add(img1, img2):
    """returns the summation of img1 and img2"""
    img1 = img1 if img1 is not None else 0
    img2 = img2 if img2 is not None else 0
    img = img1 + img2
    return img

def cosine_distance(img1,img2,w=1.0):
    """Calculates the cosine distance between img1 and imag2 and multiplies the result by a weight w """
    norm1 = np.linalg.norm(img1,axis=2)
    norm2 = np.linalg.norm(img2,axis=2)
    dot = np.sum(np.multiply(img1,img2),axis = 2)
    cos = dot/(norm1*norm2 + 1e-8)
    lamb = 1-cos

    #eliminate "ghosts" in the image represented by small angles
    lamb[np.where(lamb > 0.001)] = 0.0 

    d0 = np.absolute(norm1-norm2)
    result = (1 - lamb/2.0) * d0 * w
    return result

def euclidian_distance(img1,img2,w=1.0):
    """Calculates the euclidian distance between img1 and imag2 and multiplies the result by a weight w"""
    img = img1-img2
    euclidian = np.linalg.norm(img,axis=2)*w
    return euclidian

def get_star(frames, func = "cos", weighted = False ):
    """Return a gray scale image with the result of the star applied over all frames"""
    t = len(frames)
    
    w = [float(i)/t if weighted else 1.0 for i in range(1,t) ]
    #Star representation by cosine distance
    if func == "cos":
        return (reduce(add, map(cosine_distance, frames[:-1],frames[1:],w) ))
    #Star representation by euclidian distance
    if func == "euclidian":
        return (reduce(add, map(euclidian_distance, frames[:-1],frames[1:],w) ))
    
    #Star representation by cosine distance
    return (reduce(add, map(diff, frames[:-1],frames[1:],w)) )

    
def get_starRGB(frames, type = "cos"):
    """Return an RGB image with the result of the star applied over all frames
    :param frames: frames to claculate the star RGB representation
    :param type: the distancev to be used by star RGB function. Could be the default `cos' (cosince), `euclidian` and `None` for absolute difference.
    """
    total = len(frames) 
    step = total // 3
    r = np.expand_dims(get_star(frames[:step],type), 2)
    g = np.expand_dims(get_star(frames[step:step + (total - step*2)],type), 2)
    b = np.expand_dims(get_star(frames[step + (total - step*2):],type), 2)
    
    return (np.concatenate([r,g,b], axis = 2))


def get_star_n_channels(frames,channels):
    total = len(frames)
    w,h,c = frames[0].shape

    color_images = []
    history_images = []

    amount = [total //channels] * channels
    rest = total%channels
    pad = (channels-rest)//2
    for i in range(pad,rest+pad):amount[i] += 1
    b = 0
    for i in range(channels):
        e = b + amount[i]
        color_images.append(np.expand_dims(frames[(b+e)//2], axis=0))    
        history_images.append(np.expand_dims(normalize(get_star(frames[b:e])),axis = -1 ))
        b = e
    
    return np.concatenate(color_images, axis = 0), np.concatenate(history_images, axis = 2)
    

def get_frames(file, size = (120,160)):
    try:
        frames = pims.Video(file)
        frames = list(map(lambda f,s: m.imresize(f, tuple(s)).astype(np.float32), frames, [list(size)]*len(frames)))
        return frames
    except:
        print("Error in file = {}".format(file))
        return None




def create_mosaico(images, num_columns = 1, num_pixels_sep = 10, final_size = None):
    images = list(map(m.imresize, images, [images[0].shape]*len(images)))
    rows = len(images) // num_columns
    x,y,z = images[0].shape
    col = 0
    row = 0

    
    h, w = ( len(images)//num_columns  + (1 if len(images)%num_columns > 0 else 0) )* (x+num_pixels_sep), num_columns*(y+num_pixels_sep)
    mosaico = np.ones((h,w,z))*255
    for i,image in enumerate(images):

        row = i//num_columns*(x + num_pixels_sep)
        col = i%num_columns*(y + num_pixels_sep)

        mosaico[row:row+x,col:col+y,:] = image
    if final_size is not None:
        mosaico = m.imresize(mosaico, final_size)
    plt.imshow(np.uint8(mosaico))
    plt.show()



def get_label(name):
        #Return the label of a gesture. Name must have the following structure: <filename>_<label>.mp4
        n, _ = os.path.splitext(name)
        n = n.split("_")[-1]
        return int(n)



if __name__ == "__main__":

    frames= get_frames("Sample0437-530-559-video_13.mp4", (240,320))
    frames = np.array(frames)
    star = normalize(get_starRGB(frames))
    plt.imshow(star)
    plt.show()
    print(frames.shape)
    frames = np.transpose(frames,(1,2,0,3))
    print(frames.shape)
    star = np.uint8(255*normalize(get_starRGB(frames)))
    plt.imshow(star)
    plt.show()














