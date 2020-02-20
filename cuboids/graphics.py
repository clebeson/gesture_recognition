import numpy as np
import scipy.misc as m
import matplotlib.pyplot as plt
import glob
import math


def create_mosaico(images, num_columns = 1, num_pixels_sep = 10, final_size = None):
    images = list(map(m.imresize, images, [images[0].shape]*len(images)))
    rows = len(images) // num_columns
    x,y,z = images[0].shape
    col = 0
    row = 0

    
    h, w = ( len(images)//num_columns  + (1 if len(images)%num_columns > 0 else 0) )* (x+num_pixels_sep), num_columns*(y+num_pixels_sep)
    mosaico = np.ones((h,w,z))
    for i,image in enumerate(images):

        row = i//num_columns*(x + num_pixels_sep)
        col = i%num_columns*(y + num_pixels_sep)

        mosaico[row:row+x,col:col+y,:] = image
    if final_size is not None:
        print("sdfsdfsdfsdf")
        mosaico = m.imresize(mosaico, final_size)
    plt.imshow(np.uint8(mosaico))
    plt.show()
files = sorted(glob.glob("/home/clebeson/Pictures/att/*.png"))
images = list(map(m.imread, files))
# images = [np.zeros((30,30,3)) for _ in range(20)]
create_mosaico(images[-6:], num_columns = 2, num_pixels_sep = 2)

    















