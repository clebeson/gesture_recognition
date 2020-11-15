import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import os
import zipfile
import shutil
import cv2
import numpy
import csv
from PIL import Image, ImageDraw
from scipy.misc import imresize
import pims
import glob



class Skeleton(object):
    """ Class that represents the skeleton information """
    #define a class to encode skeleton data
    def __init__(self,data):
        """ Constructor. Reads skeleton information from given raw data """
        # Create an object from raw data
        self.joins=dict()
        pos=0
        self.joins['HipCenter']=(list(map(float,data[pos:pos+3])),list(map(float,data[pos+3:pos+7])),list(map(int,data[pos+7:pos+9])))
        pos=pos+9
        self.joins['Spine']=(list(map(float,data[pos:pos+3])),list(map(float,data[pos+3:pos+7])),list(map(int,data[pos+7:pos+9])))
        pos=pos+9
        self.joins['ShoulderCenter']=(list(map(float,data[pos:pos+3])),list(map(float,data[pos+3:pos+7])),list(map(int,data[pos+7:pos+9])))
        pos=pos+9
        self.joins['Head']=(list(map(float,data[pos:pos+3])),list(map(float,data[pos+3:pos+7])),list(map(int,data[pos+7:pos+9])))
        pos=pos+9
        self.joins['ShoulderLeft']=(list(map(float,data[pos:pos+3])),list(map(float,data[pos+3:pos+7])),list(map(int,data[pos+7:pos+9])))
        pos=pos+9
        self.joins['ElbowLeft']=(list(map(float,data[pos:pos+3])),list(map(float,data[pos+3:pos+7])),list(map(int,data[pos+7:pos+9])))
        pos=pos+9
        self.joins['WristLeft']=(list(map(float,data[pos:pos+3])),list(map(float,data[pos+3:pos+7])),list(map(int,data[pos+7:pos+9])))
        pos=pos+9
        self.joins['HandLeft']=(list(map(float,data[pos:pos+3])),list(map(float,data[pos+3:pos+7])),list(map(int,data[pos+7:pos+9])))
        pos=pos+9
        self.joins['ShoulderRight']=(list(map(float,data[pos:pos+3])),list(map(float,data[pos+3:pos+7])),list(map(int,data[pos+7:pos+9])))
        pos=pos+9
        self.joins['ElbowRight']=(list(map(float,data[pos:pos+3])),list(map(float,data[pos+3:pos+7])),list(map(int,data[pos+7:pos+9])))
        pos=pos+9
        self.joins['WristRight']=(list(map(float,data[pos:pos+3])),list(map(float,data[pos+3:pos+7])),list(map(int,data[pos+7:pos+9])))
        pos=pos+9
        self.joins['HandRight']=(list(map(float,data[pos:pos+3])),list(map(float,data[pos+3:pos+7])),list(map(int,data[pos+7:pos+9])))
        pos=pos+9
        self.joins['HipLeft']=(list(map(float,data[pos:pos+3])),list(map(float,data[pos+3:pos+7])),list(map(int,data[pos+7:pos+9])))
        pos=pos+9
        self.joins['KneeLeft']=(list(map(float,data[pos:pos+3])),list(map(float,data[pos+3:pos+7])),list(map(int,data[pos+7:pos+9])))
        pos=pos+9
        self.joins['AnkleLeft']=(list(map(float,data[pos:pos+3])),list(map(float,data[pos+3:pos+7])),list(map(int,data[pos+7:pos+9])))
        pos=pos+9
        self.joins['FootLeft']=(list(map(float,data[pos:pos+3])),list(map(float,data[pos+3:pos+7])),list(map(int,data[pos+7:pos+9])))
        pos=pos+9
        self.joins['HipRight']=(list(map(float,data[pos:pos+3])),list(map(float,data[pos+3:pos+7])),list(map(int,data[pos+7:pos+9])))
        pos=pos+9
        self.joins['KneeRight']=(list(map(float,data[pos:pos+3])),list(map(float,data[pos+3:pos+7])),list(map(int,data[pos+7:pos+9])))
        pos=pos+9
        self.joins['AnkleRight']=(list(map(float,data[pos:pos+3])),list(map(float,data[pos+3:pos+7])),list(map(int,data[pos+7:pos+9])))
        pos=pos+9
        self.joins['FootRight']=(list(map(float,data[pos:pos+3])),list(map(float,data[pos+3:pos+7])),list(map(int,data[pos+7:pos+9])))
    def getAllData(self):
        """ Return a dictionary with all the information for each skeleton node """
        return self.joins
    def getWorldCoordinates(self):
        """ Get World coordinates for each skeleton node """
        skel=dict()
        for key in self.joins.keys():
            skel[key]=self.joins[key][0]
        return skel
    def getJoinOrientations(self):
        """ Get orientations of all skeleton nodes """
        skel=dict()
        for key in self.joins.keys():
            skel[key]=self.joins[key][1]
        return skel
    def getPixelCoordinates(self):
        """ Get Pixel coordinates for each skeleton node """
        skel=dict()
        for key in self.joins.keys():
            skel[key]=self.joins[key][2]
        return skel
    def getHands(self, size):
        p = self.getPixelCoordinates()
        rh = np.array(p["HandRight"])[:2]
        lh = np.array(p["HandLeft"])[:2]

        if sum(lh) == sum(rh) == 0: return ((0,0),(0,0)), ((0,0),(0,0))
        # le = np.array(p["ElbowLeft"])
        # re = np.array(p["ElbowRight"])
        # # crarm = rh+re/2
        # clarm = lh+le/2
        # rarm = np.linalg.norm(rh-re,2)
        # larm = np.linalg.norm(lh-le,2)
        # print(rh,rh,lh,lh,"-")
        rh[0]= size if (rh[0]- size) <0 else 640 - size  if (rh[0] + size) > 640 else rh[0]
        rh[1]= size if (rh[1] -size) <0 else 480 - size if (rh[1] + size) > 480 else rh[1]

        lh[0]= size if (lh[0]  - size) <0 else 640-size if (lh[0] + size) > 640 else lh[0]
        lh[1]= size if (lh[1]  - size) <0 else 480-size if (lh[1] + size) > 480 else lh[1]
        
        return ( rh-size , rh+size ) , ( lh-size , lh+size ) 

    def toImage(self,width,height,img):
        """ Create an image for the skeleton information """
        SkeletonConnectionMap = (['HipCenter','Spine'],['Spine','ShoulderCenter'],['ShoulderCenter','Head'],['ShoulderCenter','ShoulderLeft'], \
                                 ['ShoulderLeft','ElbowLeft'],['ElbowLeft','WristLeft'],['WristLeft','HandLeft'],['ShoulderCenter','ShoulderRight'], \
                                 ['ShoulderRight','ElbowRight'],['ElbowRight','WristRight'],['WristRight','HandRight'],['HipCenter','HipRight'], \
                                 ['HipRight','KneeRight'],['KneeRight','AnkleRight'],['AnkleRight','FootRight'],['HipCenter','HipLeft'], \
                                 ['HipLeft','KneeLeft'],['KneeLeft','AnkleLeft'],['AnkleLeft','FootLeft'])
        im = Image.fromarray(img)
        draw = ImageDraw.Draw(im)
        for link in SkeletonConnectionMap:
            p=self.getPixelCoordinates()[link[1]]
            p.extend(self.getPixelCoordinates()[link[0]])
            
            draw.line(p, fill=(255,0,0), width=5)
        for node in self.getPixelCoordinates().keys():
            p=self.getPixelCoordinates()[node]
            r=5
            
            draw.ellipse((p[0]-r,p[1]-r,p[0]+r,p[1]+r),fill=(0,0,255))
        del draw
        image = numpy.array(im)
        box = 96
        (rstart,rend), (lstart,lend) = self.getHands(box//2)
        
        color = (255, 0, 0) 
        thickness = 2
        image = cv2.rectangle(image, (*rstart,), (*rend,), color, thickness)
        image = cv2.rectangle(image, (*lstart,), (*lend,), color, thickness)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
       

        return image




if __name__ == "__main__":
    # csv_files = pd.read_csv("/notebooks/datasets/Montalbano/*/samples/*_skeleton.csv").to_numpy()[1:,]
    video_files = glob.glob("/notebooks/datasets/Montalbano/*/samples/*_color.mp4")
    
    print(len(video_files))
    bbox_size = 80
    for video_file in video_files:
        csv = pd.read_csv(video_file.replace("_color.mp4","_skeleton.csv")).values[1:,]
        video = pims.Video(video_file)
        
        hand_list = np.zeros((len(video),bbox_size,2*bbox_size,3)).astype(np.uint8)
        
        lack = len(video) - len(csv)
        if lack < 0:lack=0

        for idx, (data,img) in enumerate(zip(csv,video[lack:])):
            skl = Skeleton(data)
            (rstart,rend), (lstart,lend) = skl.getHands(bbox_size//2)
            hands = np.zeros((bbox_size,2*bbox_size,3)).astype(np.uint8)
            size_rh_x = rend[1]-rstart[1]
            size_rh_y = rend[0]-rstart[0]

            size_lh_x = lend[1]-lstart[1]
            size_lh_y = lend[0]-lstart[0]
            # print(rstart,rend, lstart, lend)
            if size_lh_x > 0 and size_lh_y > 0 and size_rh_x > 0 and size_rh_y > 0:
                try:
                    # print(rstart[0],rend[0], rstart[1],rend[1], img.shape, img[lstart[1]:lend[1], lstart[0]:lend[0]].shape)
                    hands[:size_rh_x, bbox_size-size_rh_y:bbox_size]= img[rstart[1]:rend[1], rstart[0]:rend[0]]
                    hands[:size_lh_x, bbox_size:bbox_size+size_lh_y]=img[lstart[1]:lend[1], lstart[0]:lend[0]]
                except Exception as e:
                    print(video_file)
                    print(e)
                finally:
                    hand_list[idx+lack] = hands
            # img = skl.toImage(640,480,img)
            # cv2.imshow("skl",img) #hands[:,:,::-1])
            # cv2.imshow("skl",hands[:,:,::-1])
            # cv2.waitKey(10)
        np.save(video_file.replace("_color.mp4","_hands.npy"),hand_list)
        

    

