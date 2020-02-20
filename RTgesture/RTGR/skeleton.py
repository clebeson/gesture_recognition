import pandas as pd
import json
from pprint import pprint
import numpy as np
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import time
import glob


#********************************************************************************************
#The next four classes just simulate the IS-MSGS classes. They are for compatibility
class Object(): #It implements a "objects" as the "objects" attribute in ObjectAnottations
    def __init__(self, obj = None):
        if obj is not None:
            self.id = obj["id"]
            self.score = obj["score"]
            self.keypoints = []
            for kp in obj["keypoints"]:
                    self.keypoints.append(Keypoint(kp))

class Keypoint: #It implements a "keypoint" as the "keypoint" attribute in "objects"
    def __init__(self, keypoint):
            self.id = int(keypoint["id"])
            self.score = float(keypoint["score"])
            self.position = Position(keypoint["position"])

class Position: #It implements a "position" as the "position" attribute in Keypoints
    def __init__(self, position):
        self.x = position["x"]
        self.y = position["y"]
        self.z = position["z"]


class ObjectAnnotations: #Similar to ObjectAnottations
    def __init__(self, detection):
        self.frame_id = detection["frame_id"]
        self.objects = [Object(obj) for obj in detection["objects"]]

#********************************************************************************************









#------------------- Use from here --------------------------------------------

#These classes are for help the representention when implementing the recognizer
class Joint:
    def __init__(self, keypoint = None):
        if keypoint is not None:
            self.id = keypoint.id
            self.score = keypoint.score
            self.x = keypoint.position.x
            self.y = keypoint.position.y
            self.z = keypoint.position.z
        else:
            self.id = 0
            self.score = 0.0
            self.x = 0.0
            self.y = 0.0
            self.z = 0.0

    def __sub__(self, joint):
        return np.array([self.x - joint.x, self.y - joint.y, self.z - joint.z])

    def __add__(self, joint):
        return np.array([self.x + joint.x, self.y + joint.y, self.z + joint.z])

    def get3DPoint(self):
        return np.array([self.x,self.y,self.z])

        
    def __str__(self):
        return "\n   Score={}, ID={}, point = ({:.2f},{:.2f},{:.2f})".format(self.score, self.id , self.x,self.y,self.z)

class Skeleton: 
    def __init__(self, obj = None):
        if obj is not None:
            self.id = obj.id
            self.score = obj.score
            self.joints = {}
            for kp in obj.keypoints:
                    self.joints[kp.id] = Joint(kp)
        else:
            self.id = 0
            self.score = 0
            self.joints = {} 

           
    def vectorized(self):
        vec = np.ones((54))*(-1)
        for i in range(2,20):
            for key in self.joints.keys():
                j = self.joints[key]
                if j.id == i:
                    s = (i-2)*3
                    vec[s:s+3] = [j.x,j.y,j.z]
        return vec

    def vectorize_reduced(self):
        vec = np.ones((21))*(-1)
        for i,k in  enumerate([3,4,5,6,7,8,9]):
            for key in self.joints.keys():
                j = self.joints[key]
                if j.id == k:
                    s = i*3
                    vec[s:s+3] = [j.x,j.y,j.z]
        return vec

    def __str__(self):
        string = "\nSkeleton id: {}".format(self.id)
        for key in self.joints.keys():
            string += str(self.joints[key])
        return string
    
    
    def GetJoint(self, id):
        return self.joints[id] if id in self.joints.keys() else Joint()
            

    def flip(self):
        flip_keys = { 0:0, 1:1, 2:2, 3:3, 7:4, 
                      8:5, 9:6, 4:7, 5:8, 6:9,
                      13:10,14:11,15:12,10:13,
                      11:14,12:15,17:16,16:17,
                      19:18,18:19,20:20
        }

        
        skl = Skeleton()
        skl.id= self.id
        skl.score = self.score
        for key in self.joints.keys():
            skl.joints[flip_keys[key]] = self.joints[key]

        return skl


    def normalize(self):
        rh = self.GetJoint(10)
        ls = self.GetJoint(7)
        mean = (rh+ls)/2
        center = Joint()
        center.x = mean[0]
        center.y = mean[1]
        center.z = mean[2]
        skl = Skeleton()
        skl.id= self.id
        skl.score = self.score

        for key in self.joints.keys():
            point  = self.joints[key] - center
            joint = Joint()
            joint.id =  self.joints[key].id
            joint.score =  self.joints[key].score
            joint.x = point[0]
            joint.y = point[1]
            joint.z = point[2]
            skl.joints[key] = joint

        return skl





    def __str__(self):
        string = "\nFrame_id = {}\n".format(self.frame_id)
        for skl in self.skeletons:
            string += str(skl)
        return string
        

 
# It Will update the buffer with the skeletons present on te current frame
def update_buffer(skeletons, buffer, window_size):
    for i, skl in enumerate(skeletons):
        if i >= max_skl_in_scene: break
        if len(buffer[i]) >= window_size:
            buffer[i].pop(0)

        buffer[i].append(skl)


def save_skeletons():
    json_files = glob.glob("/public/datasets/ufes-2020-01-23/*3d.json") 
    # json_files = glob.glob("/public/datasets/ifes-2018-10-19/*3d.json") 
    
    # size = len(json_files)
    # test_files = json_files[int(size*0.8):]
    # json_files = json_files[:int(size*0.8)]
    save(json_files)
    #save(test_files,"test")

def save(files, file_name = "ufes_dataset"):
    print(len(files))
    dataset = []
    dataset_flip = []
    for src in files:
        name = src.split("/")[-1].split("_")[0]
        spots = glob.glob("/public/datasets/ufes-2020-01-23/{}_spots.json".format(name))[0]
        with open(src) as f:
            data = json.load(f)
        with open(spots) as f:
            spots = json.load(f)
        labels = np.zeros((int(spots["n_samples"])))
        spots = spots["labels"]
        label = float(name.split("g")[-1])
        for spot in spots:
            b,e = spot["begin"], spot["end"]
            labels[b:e] = label

        for i, localization in enumerate(data["localizations"]):
            annotations = ObjectAnnotations(localization)  
            skeletons = [Skeleton(obj) for obj in annotations.objects]
            for skl in skeletons:
                skl_normalized = skl.normalize()
                dataset.append(np.append(np.array([labels[i]]),skl_normalized.vectorize_reduced(),axis=0))
                # skl_flip_normalized = skl.flip().normalize()
                # dataset_flip.append(np.append(np.array([labels[i]]),skl_flip_normalized.vectorize_reduced(),axis=0))
                break
    np.save(file_name,np.array(dataset+dataset_flip))
       

   
#Try to recognize the wave gesture
def recognize_wave_gesture(buffer): 
    
    for id_skl, skletons in enumerate(buffer):
        satisfied = [] #it will accumulate 0 or 1 depending whether the wave condition was satisfied in a specific skeleton
        
        if len(skletons) < window_size: continue # The recpgnize will be perfrmed just when the buffer is full
        
        for skl in skletons:
            if skl == []: continue
            
            #normalize joints using the left_hip point as referential
            skl_normalized = skl.normalize()
            
            #Getting the joints of interest
            neck = skl_normalized.GetJoint(3) 
            left_hand = skl_normalized.GetJoint(9)
            right_hand = skl_normalized.GetJoint(6)
            
            #if the condition is satisfied with the right or left hand, the "satisfied" variable will receive 1, if is not, 0
            if right_hand.z > neck.z:
                diff = np.power(right_hand  - neck, 2)
                dist_to_neck = np.sqrt([ diff[0] + diff[1] ])
                if dist_to_neck < threshold:
                    satisfied.append(1.0)

            elif left_hand.z > neck.z:
                diff = np.power(left_hand  - neck, 2)
                dist_to_neck = np.sqrt([diff[0] + diff[1]])
                if dist_to_neck < threshold:
                    satisfied.append(1.0)
            else:
                satisfied.append(0.0)

        #buffer[id_skl] = buffer[int(delay):] #This is impleenting the delay
        if np.mean(satisfied) > satisfied_percentage: #Thare is a chance of this threshold need to be changed. 
            buffer[id_skl] = []#This is impleenting the delay

            return skl

    return None


def plot_skeleton(ax, fig, skeleton):
    ax.clear()
    ax.set_xlim(-1.0,1.0)
    ax.set_ylim(-1.0,1.0)
    ax.set_zlim(0,2.0)
    points = np.array([skeleton.joints[key].get3DPoint() for key in skeleton.joints.keys()])
    ax.scatter(points[:,0], points[:,1], points[:,2])
    fig.canvas.draw()

    
def perform_task(skeleton):
    pass #To do


if __name__ == "__main__":
   save_skeletons()
    
   
#     #-------------- Hyperparameters --------------------------------
#     fps = 10.0
#     gesture_execution_time = 2.0 #duration of the gesture in seconds
#     max_skl_in_scene = 10 # Number max of skeletons to be detected in a scene
#     threshold = 0.8 #it will impose a condition at the distance between the neck and a hand (left or right). 
#                     #It is necessary when the user user do a point gesture that is upper then a shouder but it is far from the neck.
#     satisfied_percentage = 0.4 #It is measuring how many satisfied skeletons occured in the time window
#    #-----------------------------------------------------------------
   
#     window_size =  gesture_execution_time * fps
#     buffer = [ [] for _ in range(max_skl_in_scene)]
   
   
   
   
#    # Just for plotting the skeleton in my debug
#     plt.ion()
#     fig = plt.figure()
#     ax = fig.gca(projection='3d')
    



#     # In this case the skeletons are reading fram a JSON file. But it can be read from the IS.
#     json_files = glob.glob("./ask_for_help/p001g01_3d.json")
#     for src in json_files:
#         print(src)
#         with open(src) as f:
#             data = json.load(f)
#         for i, localization in enumerate(data["localizations"]):
            
#             annotations = ObjectAnnotations(localization)  
#             skeletons = [Skeleton(obj) for obj in annotations.objects]
                   
#             #plot all skeletons
#             for skl in skeletons:
#                 plot_skeleton(ax, fig, skl)  
            
            
#             update_buffer(skeletons, buffer, window_size) # Here shoud be passed a list of skeletons detected on a image
#             wave_skeleton = recognize_wave_gesture(buffer) # test if the Wave was recognazed. If it's true, it will return the last skeleton such composed the gesture, 
#                                                                                         #else it willreturn Non
#             if wave_skeleton is not None: 
#                 perform_task(wave_skeleton) # The function which will send the command to the pepper path planning controller
                
#                 #releasing buffer
#                 del buffer
#                 buffer = [ [] for _ in range(max_skl_in_scene)]
               
#                 #Just for debugging
#                 print("recognized = ", i) 
                
#             time.sleep(fps * 0.001) #Trying to simulate a real-time environment 


    
    












