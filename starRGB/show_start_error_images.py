import imageio
imageio.plugins.ffmpeg.download()
from random import shuffle as sf
import numpy as np
import matplotlib.pyplot as plt
import pims
import os
import logging
import glob
import cv2
import scipy.stats as st
import itertools
import matplotlib.pyplot as plt
from skimage.transform import resize
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from scipy import misc
from scipy.misc import imread, imresize
from scipy.ndimage.filters import gaussian_filter


#import conv_cosnorm as cos
#59% in test

#os.system("pip install pims")

def plot_images(images, titles, subplot = (1,2), show_size=100):
    if not images or len(images) == 0:
        return
    """
    The show_size is the number of pixels to show for each image.
    The max value is 299.
    """
    from skimage.transform import resize
    def normalize_image(x):
        x_min = x.min()
        x_max = x.max()
        x_norm = (x - x_min) / (x_max - x_min)
        return x_norm

    # Create figure with sub-plots.

    fig, axes = plt.subplots(*subplot)

    # Adjust vertical spacing.
    fig.subplots_adjust(hspace=1, wspace=0.01)

    # Use interpolation to smooth pixels?
    smooth = True

    # Interpolation type.
    if smooth:
        interpolation = 'spline16'
    else:
        interpolation = 'nearest'

    # For each entry in the grid.
    size = len(images)
    white = np.uint8(np.zeros_like(images[0])+255)
    white = img = resize(white, (show_size, show_size))
   
    for i, ax in enumerate(axes.flat):
        if i >= size:
           img_norm = white  
           title = "" 
        # Get the i'th image and only use the desired pixels.
        else:
            img = images[i]
            img = resize(img, (show_size, show_size))
            title = titles[i]

            # Normalize the image so its pixels are between 0.0 and 1.0
            img_norm = normalize_image(img)

        # Plot the image.
        ax.imshow(img_norm, interpolation=interpolation)
        ax.set_title(title)

        # Remove ticks.
        ax.set_xticks([])
        ax.set_yticks([])

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()

files = glob.glob("/home/clebeson/attention_maps/*.jpg")
class_names = ['prendere', 'vieniqui', 'perfetto', 'fame', 'sonostufo', 'seipazzo', 'basta', 'cheduepalle', 'noncenepiu', 'chevuoi',
                'ok', 'combinato', 'freganiente', 'cosatifarei', 'buonissimo', 'vattene', 'messidaccordo', 'daccordo', 'furbo', 'tantotempo']
images = []
for cn in class_names:
    
    names = [file for file in files if file.split("/")[-1].startswith(cn)]
    names = [name for name in names if "_img" in name and "gd_img" not in name]
    labels = [label.split("_")[2] for label in names]
    images = [imread(file) for file in names ]
    print(cn, len(images))
    #plot_images(images, labels, subplot = (6,6), show_size=100)
     