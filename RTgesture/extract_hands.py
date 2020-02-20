from skeleton import *
import numpy as np
import matplotlib.pyplot as plt 
import cv2
import pims
import math
import scipy.misc as m
import glob
import time
from skin import *

def get_frames(file):
    frames = pims.Video(file)
   
    return frames


def handDetect(skeleton, image, out_size = (240,320)):
    # right hand: wrist 4, elbow 3, shoulder 2
    # left hand: wrist 7, elbow 6, shoulder 5
    images = np.zeros((2,out_size[0],out_size[1],3))
    ratioWristElbow = 0.5
    image_height, image_width = image.shape[0:2]
    
    joints = skeleton.joints

    # if any of three not detected
    
    left_shoulder = skeleton.GetJoint(7)
    left_elbow  = skeleton.GetJoint(8)
    left_wrist = skeleton.GetJoint(9)
    right_shoulder  = skeleton.GetJoint(4)
    right_elbow  = skeleton.GetJoint(5)
    right_wrist = skeleton.GetJoint(6)
    score = 0.6
    has_left = ( (left_shoulder.get3DPoint().sum() !=0 and left_shoulder.score > score) and \
             (left_elbow.get3DPoint().sum() !=0 and left_elbow.score > score) and \
             (left_wrist.get3DPoint().sum() !=0 and left_elbow.score > score) \
             )
    has_right =  ( (right_shoulder.get3DPoint().sum() !=0 and right_shoulder.score > score) and \
             (right_elbow.get3DPoint().sum() !=0 and right_elbow.score > score) and \
             (right_wrist.get3DPoint().sum() !=0 and right_elbow.score > score) \
             )
    
        
    hands = []
    #left hand
    if has_left:
       
        x1, y1, _ = left_shoulder.get3DPoint()
        x2, y2, _ = left_elbow.get3DPoint()
        x3, y3, _ = left_wrist.get3DPoint()
        hands.append([x1, y1, x2, y2, x3, y3, True])
    
    # right hand
    if has_right:
        x1, y1, _ = right_shoulder.get3DPoint()
        x2, y2, _ = right_elbow.get3DPoint()
        x3, y3, _ = right_wrist.get3DPoint()
        hands.append([x1, y1, x2, y2, x3, y3, False])

    for x1, y1, x2, y2, x3, y3, is_left in hands:
        x = x3 + ratioWristElbow * (x3 - x2)
        y = y3 + ratioWristElbow * (y3 - y2)
        distanceWristElbow = math.sqrt((x3 - x2) ** 2 + (y3 - y2) ** 2)
        distanceElbowShoulder = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        width = 1 * max(distanceWristElbow, 0.9 * distanceElbowShoulder)

        x -= width / 2
        y -= width / 2  # width = height
        # overflow the image
        if x < 0: x = 0
        if y < 0: y = 0
        width1 = width
        width2 = width
        if x + width > image_width: width1 = image_width - x
        if y + width > image_height: width2 = image_height - y
        width = min(width1, width2)
        # the min hand box value is 10 pixels
        if width >= 10:
            x =int(x)
            y=int(y)
            w = int(width)
            img = image[y:y+w, x:x+w,:]
            img = m.imresize(img, out_size)
            if is_left: images[0] = img
            else: images[1] = img
       
    return np.uint8(images)

if __name__ == "__main__":
    files = glob.glob("ifes_samples/*_3d.json")
    names = [file.replace("_3d.json", "") for file in files]
    cams = ["c00", "c01","c02","c03"]
    out_size = (110,120)
    for name in names:
        videos = []
        files2d = []
        
        for file in [ "{}{}_2d.json".format(name,cam) for cam in cams]:
            with open(file) as f:
                files2d.append(iter(json.load(f)["annotations"]))
        
        for file in ["{}{}.mp4".format(name,cam) for cam in cams]:
            videos.append(get_frames(file) )

        size = len(videos[0])
        for sample in range(size):
            hands = [[],[]]
           # t = time.time()
            for i in range(4):
                has_hands = False
                img = videos[i][sample]
                localization = next(files2d[i])
                annotations = ObjectAnnotations(localization) 
                skeletons = [Skeleton(obj) for obj in annotations.objects]
                # if i == 2: 
                #     print(len(skeletons))
                #     cv2.imshow("image",img)
                    # cv2.waitKey(30)
                for skl in skeletons:
                    joint = skl.GetJoint(10) 
                    if (joint.x >250 and joint.x < 900) and (joint.y>275 and joint.y<565):
                        hand = handDetect(skl,img, out_size)
                        #hand = [np.uint8(extractSkin(h)) for h in hand]
                        hands[0].append(hand[0])
                        hands[1].append(hand[1])
                        has_hands= True
                        break
                if not has_hands:
                    size = out_size+(3,)
                    hands[0].append(np.zeros(size))
                    hands[1].append(np.zeros(size))
            
            hands = np.vstack([np.hstack(hand) for hand in hands] )
            #print("elapsed = ",time.time() - t)
            
            cv2.imshow("hand",hands[:,:,[2,1,0]])
            #cv2.imshow("image", img)
            cv2.waitKey(100)
                        
