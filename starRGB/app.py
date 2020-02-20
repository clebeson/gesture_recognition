from ChalearnLAPSample import GestureSample
import cv2
from matplotlib import pylab
import datetime
import numpy as np
from scipy import ndimage, misc
import os
from itertools import izip
from PIL import Image
import csv
from moviepy.editor import *

source_paths=['/home/clebeson/validation']
dest_paths=['/home/clebeson/validation/RGB']

# def transform_rgb(gestureSample, shape_out):
#         numFrames=gestureSample.getNumFrames()
#         if numFrames > 3:
#             gestures=gestureSample.getGestures()
#             for gesture in gestures:
#                 label = gesture[0]
#                 begin = gesture[1]
#                 end   = gesture[2]
#                 count_files[label-1] += 1
#         else:
#             print '*****************************************************'
def getGestures(sample):
    labelsPath=os.path.join(source_path,'labels',sample + '_prediction.csv')
    if not os.path.exists(labelsPath):
       # warnings.warn("Labels are not available", Warning)
        return []           
    else:
        labels=[]
        with open(labelsPath, 'rb') as csvfile:
            filereader = csv.reader(csvfile, delimiter=',')
            for row in filereader:
                labels.append(map(int,row))
            del filereader
        return labels
        
def get_rgb(gestureSample, sample, shape_out, dest_path):
        gestures = getGestures(sample)
        for gesture in gestures:        
            label = gesture[0]
            begin = gesture[1]
            end   = gesture[2]
            before = datetime.datetime.now()
            images = []
            image = []
            for frame_id in range(begin+1, end):
                image = cv2.resize(gestureSample.getRGB(frame_id),shape_out, interpolation = cv2.INTER_CUBIC)
                images.append( image )
                name = "{}/{}_{}_{}_{}.mp4".format(dest_paths[0], sample, begin, end,label)
            star_clip = ImageSequenceClip(images, fps=20.0)
            star_clip.write_videofile(name)        

                
        




def transform_rgb(gestureSample, shape_out, dest_path):
        total_images = np.sum(count_files)
        numFrames=gestureSample.getNumFrames()
        if numFrames > 3:
            gestures = gestureSample.getGestures()
            image_result = np.ndarray(shape=(shape_out[0],shape_out[1], 3),
                            dtype=np.float32)
            
            for gesture in gestures:
                
                label = gesture[0]
                begin = gesture[1]
                end   = gesture[2]
                count = 1
                img_star = np.zeros((shape_out,shape_out), dtype=np.float32)
                num_images = int((end-begin)/3)
                img_prev = cv2.cvtColor(gestureSample.getRGB(1),cv2.COLOR_RGB2GRAY)
                img_prev =  cv2.resize(img_prev,(shape_out,shape_out), interpolation = cv2.INTER_CUBIC).astype(np.float32)
                
                before = datetime.datetime.now()

                for frame_id in range(begin+1, end):
                    
                    gray= cv2.cvtColor(gestureSample.getRGB(frame_id), cv2.COLOR_RGB2GRAY )
                    img_now =  cv2.resize(gray,(shape_out,shape_out), interpolation = cv2.INTER_CUBIC).astype(np.float32)
                         
                    img_star += np.absolute( img_now - img_prev )

                    img_prev = img_now
                     
                    if count == num_images:
                        max = np.nanmax(img_star)
                        min = np.nanmin(img_star)
                        image_result[:,:, 0] = 255.0 * ((img_star - min) / (max - min)) 
                        img_star = np.zeros((shape_out,shape_out), dtype=np.float32)

                    elif count == (2*num_images):
                        max = np.nanmax(img_star)
                        min = np.nanmin(img_star)
                        image_result[:,:, 1] = 255.0 * ((img_star - min) / (max - min)) 
                        img_star = np.zeros((shape_out,shape_out), dtype=np.float32)


                    count += 1

                
                max = np.nanmax(img_star)
                min = np.nanmin(img_star)
                image_result[:,:, 2] = 255.0 * ((img_star - min) / (max - min)) 

                # cv2.imshow('R', image_result.astype(np.uint8)[:, :, 0])
                # cv2.imshow('G', image_result.astype(np.uint8)[:, :, 1])
                # cv2.imshow('B', image_result.astype(np.uint8)[:, :, 2])

                # cv2.waitKey(2000)
                

                name = gestureSample.getGestureName(label)
                count_files[label-1] += 1

                file_result = '{}.png'.format(os.path.join(dest_path,name+'_'+str(count_files[int(label-1.0)])))
                misc.imsave(file_result, image_result )
                after = datetime.datetime.now()
                took = after - before
                total_images += 1
                print('Save at: {} \nTook: {} ms \nSaved Images: {}'.format(file_result, took.total_seconds()*1000, total_images))
        
if __name__ == '__main__' :
    for source_path, dest_path in izip(source_paths,dest_paths):
        for database in os.listdir(source_path):
            if database !='.' or database !='..' or not os.path.isfile(database):
                for sample in  os.listdir(os.path.join(source_path,database)):
                    file_name = os.path.join(source_path,database,sample)                
                    if sample != '.' and sample != '..' and not os.path.isfile(file_name):
                        gestureSample = GestureSample(file_name)
                        get_rgb(gestureSample,sample, (320,240), dest_path)
                        del gestureSample

# gestureSample = GestureSample('/home/clebeson/Documents/Data_base/data/Train1/Sample0001')



