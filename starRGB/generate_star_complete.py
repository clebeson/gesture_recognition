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
    """Calculates absolute diffderence between  img1 and imag2 and multiplies by a weight w """
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

    lamb[np.where(lamb > 0.001)] = 0.0
    d0 = np.absolute(norm1-norm2)
    result = (1 - lamb/2.0) * d0 * w
    return result

def euclidian_distance(img1,img2,w):
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
    #Star representation by euclidian distnace
    if func == "euclidian":
        return (reduce(add, map(euclidian_distance, frames[:-1],frames[1:],w) ))
    
    #Star representation by cosine distance
    return (reduce(add, map(diff, frames[:-1],frames[1:],w)) )

    
def get_starRGB(frames, type = "cos"):
    """Return a RGB image with the result of the star applied over all frames
    :param frames: frames to claculate the star RGB representation
    :param type: the distancev to be used by star RGB function. Could be the default `cos' (cosince), `euclidian` and `None` for absolute difference.
    """
    total = len(frames) 
    step = total // 3
    r = np.expand_dims(get_star(frames[:step],type), 2)
    g = np.expand_dims(get_star(frames[step:step*2],type), 2)
    b = np.expand_dims(get_star(frames[step*2:],type), 2)
    
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
    
    # if len(color_images) < channels: 
    #     zero_image = np.zeros((w,h,1))
    #     while len(color_images) < channels: 
    #         color_images.append(zero_image)
    #         history_images.append(zero_image)

    #     del zero_image
    return np.concatenate(color_images, axis = 0), np.concatenate(history_images, axis = 2)
    

def get_frames(file, size = (120,160)):
    try:
        frames = pims.Video(file)
        frames = list(map(lambda f,s: m.imresize(f, tuple(s)).astype(np.float32), frames, [list(size)]*len(frames)))
        return frames
    except:
        print("Error in file = {}".format(file))
        return None


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
    n = n.split("-")[-1]
    return int(n)


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
    star = normalize(get_starRGB(frames))
    plt.imshow(star)
    plt.show()




# images = [np.zeros((30,30,3)) for _ in range(20)]
# create_mosaico(images, num_columns = 6, num_pixels_sep = 0)

# print("\n\n","*"*20,"\n")
# files = glob.glob("./test_videos/*.mp4")
# print("files {}".format(len(files)))
# channels = 9
# total =len(files)

# color = np.zeros((total,channels,120,160,3))
# history = np.zeros((total,120,160,channels))

# labels = np.zeros(total)
# total_images = 0

# for i, file in enumerate(files):
#     frames= get_frames(file, (120,160))
     
#     if frames is None: continue

#     total_images += len(frames)
#     if len(frames) < (channels*2)+1:  print(len(frames))
#     while len(frames) < (channels*2)+1 : frames.append(frames[-1])
#     c,h = get_star_n_channels(frames, channels)
   
    
#     n, _ = os.path.splitext(file.split("/")[-1])
    
#     try:
#         os.makedirs("./test_videos/{}".format(n) )
#     except: pass
    


#     m.imsave("./test_videos/{}/history_0.png".format(n), np.uint8(h[:,:,:3] * 255.0))
#     m.imsave("./test_videos/{}/history_1.png".format(n), np.uint8(h[:,:,3:6] * 255.0))
#     m.imsave("./test_videos/{}/history_2.png".format(n), np.uint8(h[:,:,6:9] * 255.0))

#     for j in range(9):
#         m.imsave("./test_videos/{}/color_{}.png".format(n,j), np.uint8(c[j]))
#     if i % 100 == 0: print("Image {}/{}".format(i,total))
# print("total frames:",total_images)



# print("\n\n","*"*20,"\n")
# files = glob.glob("./train_videos/*.mp4")
# print("files {}".format(len(files)))
# channels = 10
# total =len(files)

# color = np.zeros((total,channels,120,160,3))
# history = np.zeros((total,120,160,channels))

# labels = np.zeros(total)
# total_images = 0

# for i, file in enumerate(files[:200]):
#     frames= get_frames(file, (120,160))

#     if frames is None: continue

#     total_images += len(frames)
#     if len(frames) < (channels*2)+1:  print(len(frames))
#     while len(frames) < (channels*2)+1 : frames.append(frames[-1])
#     c,h = get_star_n_channels(frames, channels)
#     # color[i] = c
#     # history[i] = normalize(h)
#     # labels[i] = get_label(file)
#     if i % 100 == 0: print("Image {}/{}".format(i,total))
#     n, _ = os.path.splitext(file.split("/")[-1])

#     np.savez("star_n_channels/{}".format(n),color=c, history = h, labels=get_label(file))
#     print("total frames:",total_images)
# del images, labels, history




















# file= "./Sample0004_color_163_200_13.mp4"
# frames= get_frames(file, (120,160))
# cos = normalize(get_star(frames))
# eu = normalize(get_star(frames, "euclidian"))
# # frames_gray = list(map(cv2.cvtColor, frames, [cv2.COLOR_RGB2GRAY]*len(frames)))
# star = normalize(get_star(frames, "star"))
# m.imsave("cos.png", np.uint8(cos*255))
# m.imsave("eu.png", np.uint8(eu*255))
# m.imsave("star.png", np.uint8(star*255))

# m.imsave("cos-eu.png", np.uint8(normalize(cos-eu)*255))
# m.imsave("cos-star.png", np.uint8(normalize(cos-star)*255))
# m.imsave("star-eu.png", np.uint8(normalize(star-eu)*255))

# cos1,star1 = cos, star
# cos = normalize(get_star(frames[::-1]))
# eu = normalize(get_star(frames[::-1], "euclidian"))
# # frames_gray = list(map(cv2.cvtColor, frames[::-1], [cv2.COLOR_RGB2GRAY]*len(frames)))
# star = normalize(get_star(frames[::-1], "star"))
# m.imsave("cos_oposite.png", np.uint8(cos*255))
# m.imsave("eu_oposite.png", np.uint8(eu*255))
# m.imsave("star_oposite.png", np.uint8(star*255))

# m.imsave("cos-eu_oposite.png", np.uint8(normalize(cos-eu)*255))
# m.imsave("cos-star_oposite.png", np.uint8(normalize(cos-star)*255))
# m.imsave("star-eu_oposite.png", np.uint8(normalize(star-eu)*255))
# print(np.min(normalize(normalize(cos1) -normalize(cos))), np.max(normalize(normalize(cos1) -normalize(cos))))
# m.imsave("cos-cos_oposite.png", np.uint8(normalize(normalize(cos1) -normalize(cos))*255))
# m.imsave("star-star_oposite.png", np.uint8(normalize(star1-star)*255))



# cos = normalize(get_star_1_RGB(frames))
# eu = normalize(get_star_1_RGB(frames, "euclidian"))
# star = normalize(get_star_1_RGB(frames, "star"))
# m.imsave("cos_RGB.png", np.uint8(cos*255))
# m.imsave("eu_RGB.png", np.uint8(eu*255))
# m.imsave("star_RGB.png", np.uint8(star*255))

# m.imsave("cos_RGB-eu.png", np.uint8(normalize(cos-eu)*255))
# m.imsave("cos_RGB-star.png", np.uint8(normalize(cos-star)*255))
# m.imsave("star_RGB-eu.png", np.uint8(normalize(star-eu)*255))

# cos1,star1 = cos, star
# cos = normalize(get_star_1_RGB(frames[::-1]))
# eu = normalize(get_star_1_RGB(frames[::-1], "euclidian"))
# star = normalize(get_star_1_RGB(frames[::-1], "star"))
# m.imsave("cos_RGB_oposite.png", np.uint8(cos*255))
# m.imsave("eu_RGB_oposite.png", np.uint8(eu*255))
# m.imsave("starRGB_oposite.png", np.uint8(star*255))

# m.imsave("cos_RGB-eu_oposite.png", np.uint8(normalize(cos-eu)*255))
# m.imsave("cos_RGB-star_oposite.png", np.uint8(normalize(cos-star)*255))
# m.imsave("star_RGB-eu_oposite.png", np.uint8(normalize(star-eu)*255))

# m.imsave("cos-cos_oposite_RGB.png", np.uint8(normalize(normalize(cos1) -normalize(cos))*255))
# m.imsave("star-star_oposite_RGB.png", np.uint8(normalize(star1-star)*255))
