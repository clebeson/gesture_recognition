from __future__ import print_function
import numpy as np 
import scipy.ndimage as ndi 
import os 
import csv
import pims
from functools import reduce
from moviepy.editor import *
import scipy.misc as m
from scipy import ndimage
import glob
from scipy.stats import norm
import matplotlib.pyplot as plt
import pickle
from  scipy.spatial.distance import cosine
import imageio
imageio.plugins.ffmpeg.download()


def normalize(image):
    """Normalizes image between 0 and 255"""
    max = np.max(image[:])
    min = np.min(image[:])
    image = (image-min) / (max-min+1.0) 
    return np.uint8(image*255)



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
    n = len(frames)
    
    w = [float(i)/n if weighted else 1.0 for i in range(1,n) ]
    #Star representation by cosine distance
    if func == "cos":
        return (reduce(add, map(cosine_distance, frames[:-1],frames[1:],w) ))
    #Star representation by euclidian distance
    if func == "euclidian":
        return (reduce(add, map(euclidian_distance, frames[:-1],frames[1:],w) ))
    
    #Star representation using only absolute difference
    return (reduce(add, map(diff, frames[:-1],frames[1:],w)) )

    
def get_starRGB(frames, type = "cos",weighted = False):
    """Return an RGB image with the result of the star applied over all frames
    :param frames: frames to claculate the star RGB representation
    :param type: the distancev to be used by star RGB function. Could be the default `cos' (cosince), `euclidian` and `None` for absolute difference.
    """
    total = len(frames) 
    step = total // 3
    r = np.expand_dims(get_star(frames[:step],type, weighted), 2)
    g = np.expand_dims(get_star(frames[step:step + (total - step*2)],type, weighted), 2)
    b = np.expand_dims(get_star(frames[step + (total - step*2):],type, weighted), 2)
    
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
    # frames= np.array(get_frames("samples/Sample0437-530-559-video_13.mp4", (240,320)))
    # star = (get_star(frames,func="diff", weighted=False))
    # x = normalize(ndimage.sobel(star,0))
    # y = normalize(ndimage.sobel(star,1))
    # start = normalize(star)
    # m.imsave("dx.png",x)
    # m.imsave("dy.png",y)
    # m.imsave("star_diff.png",star)
   

    # star = np.concatenate([star,x,y],1)
    # plt.imshow(star, cmap='gray')
    # plt.show()


 
    for dataset in ["train","test", "validation"]:
        files = glob.glob("/home/clebeson/experiments/datasets/Montalbano/{}/videos/*.mp4".format(dataset))
        print(len(files))
        for video in files:
            print(video)
            frames= np.array(get_frames(video, (240,320)))
            # starRGB = normalize(get_starRGB(frames,type="cos", weighted=False))
            # starRGB_w = normalize(get_starRGB(frames,type="cos", weighted=True))

            # starRGBdiff = normalize(get_starRGB(frames,type="diff", weighted=False))
            # starRGBdiff_w = normalize(get_starRGB(frames,type="diff", weighted=True))

            # star = normalize(get_star(frames,func="diff", weighted=False))
            star = (get_star(frames,func="diff", weighted=False))
            x = normalize(ndimage.sobel(star,0))
            y = normalize(ndimage.sobel(star,1))
            start = normalize(star)
            star = np.stack([star,x,y],2)


            star_w = (get_star(frames,func="diff", weighted=True))
            # star_w = np.stack([star_w,star_w,star_w],2)
            x = normalize(ndimage.sobel(star_w,0))
            y = normalize(ndimage.sobel(star_w,1))
            star_w = normalize(star_w)
            star_w = np.stack([star_w,x,y],2)

            m.imsave(video.replace("videos","images").replace(".mp4",".png").replace("Sample","starsobel_Sample"),star)
            m.imsave(video.replace("videos","images").replace(".mp4",".png").replace("Sample","starsobelW_Sample"),star_w)
            # m.imsave(video.replace("videos","images").replace(".mp4",".png").replace("Sample","starRGBdiff_Sample"),starRGBdiff)
            # m.imsave(video.replace("videos","images").replace(".mp4",".png").replace("Sample","starRGBdiffW_Sample"),starRGBdiff_w)
            # m.imsave(video.replace("videos","images").replace(".mp4",".png").replace("Sample","star_Sample"),star)
            # m.imsave(video.replace("videos","images").replace(".mp4",".png").replace("Sample","starW_Sample"),star_w)

            # imgR = normalize(get_starRGB(frames[:,:,:,0],type="diff", weighted=False))
            # imgG = normalize(get_starRGB(frames[:,:,:,1],type="diff", weighted=False))
            # imgB = normalize(get_starRGB(frames[:,:,:,2],type="diff", weighted=False))
            # img = np.array([imgR,imgG, imgB])

            # imgRW = normalize(get_starRGB(frames[:,:,:,0],type="diff", weighted=True))
            # imgGW = normalize(get_starRGB(frames[:,:,:,1],type="diff", weighted=True))
            # imgBW = normalize(get_starRGB(frames[:,:,:,2],type="diff", weighted=True))
            # imgW = np.array([imgRW,imgGW, imgBW])


            # np.save(video.replace("videos","images").replace(".mp4",".npy").replace("Sample","starRGB3_Sample"), img)
            # np.save(video.replace("videos","images").replace(".mp4",".npy").replace("Sample","starRGB3W_Sample"), imgW)


            


           
   














