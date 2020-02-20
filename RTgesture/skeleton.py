import pandas as pd
import json
from pprint import pprint
import numpy as np
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import pickle
import time
import glob



JOINTS = {
    "UNKNOWN_HUMAN_KEYPOINT":0,
    "HEAD":1,
    "NOSE":2,
    "NECK":3,
    "RIGHT_SHOULDER":4,
    "RIGHT_ELBOW":5,
    "RIGHT_WRIST":6,
    "LEFT_SHOULDER":7,
    "LEFT_ELBOW":8,
    "LEFT_WRIST":9,
    "RIGHT_HIP":10,
    "RIGHT_KNEE":11,
    "RIGHT_ANKLE":12,
    "LEFT_HIP":13,
    "LEFT_KNEE":14,
    "LEFT_ANKLE":15,
    "RIGHT_EYE":16,
    "LEFT_EYE":17,
    "RIGHT_EAR":18,
    "LEFT_EAR":19,
    "CHEST":20,
    "TORSO":21
}


JOINTS_DIST = {
    "HEAD":1,
    "NOSE":2,
    "NECK":3,
    "RIGHT_SHOULDER":4,
    "LEFT_SHOULDER":7,
    "RIGHT_HIP":10,
    "RIGHT_KNEE":11,
    "LEFT_HIP":13,
    "LEFT_KNEE":14,
    "TORSO":21,
}


LINKS = [

        ('TORSO','LEFT_HIP'),
        ('TORSO', 'NECK'),
        ('TORSO', 'RIGHT_HIP'),

         #('NECK','HEAD'), 
         ('NECK', 'LEFT_SHOULDER'),
         ('NECK', 'RIGHT_SHOULDER'),
         ('NECK', 'NOSE'),
         
         ('LEFT_SHOULDER', 'LEFT_ELBOW'),
         ('RIGHT_SHOULDER', 'RIGHT_ELBOW'),
         ('LEFT_HIP','LEFT_KNEE'),
         ('RIGHT_HIP', 'RIGHT_KNEE'),

        ('NOSE', 'LEFT_EYE'),
        ('NOSE', 'RIGHT_EYE'),


         ('LEFT_ELBOW', 'LEFT_WRIST'),
         ('RIGHT_ELBOW', 'RIGHT_WRIST'),
         ('LEFT_KNEE', 'LEFT_ANKLE'),
         ('RIGHT_KNEE', 'RIGHT_ANKLE'),
         
        ('LEFT_EYE','LEFT_EAR'),
        ('RIGHT_EYE', 'RIGHT_EAR'),
         ]

CONNECTED_JOINTS = [
    ('HEAD', 'NECK', 'LEFT_SHOULDER'),
    ('HEAD', 'NECK', 'RIGHT_SHOULDER'),
    ('NECK', 'LEFT_SHOULDER', 'LEFT_ELBOW'),
    ('NECK', 'RIGHT_SHOULDER', 'RIGHT_ELBOW'),
    ('LEFT_SHOULDER', 'LEFT_ELBOW', 'LEFT_WRIST'),
    ('RIGHT_SHOULDER', 'RIGHT_ELBOW', 'RIGHT_WRIST'),
    ('LEFT_SHOULDER', 'NECK', 'LEFT_HIP'),
    ('RIGHT_SHOULDER', 'NECK', 'RIGHT_HIP'),
    ('LEFT_HIP', 'NECK', 'RIGHT_HIP'),
    ('LEFT_SHOULDER', 'NECK', 'RIGHT_SHOULDER'),
    # virtual angles
    ('LEFT_ELBOW', 'LEFT_WRIST', 'TORSO'),
    ('RIGHT_ELBOW', 'RIGHT_WRIST', 'TORSO'),
]

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
            self.x = -1
            self.y = -1
            self.z = -1

    def set_3DPoint(self, point):
        self.x,self.y,self.z = point

    def __sub__(self, joint):
        return np.array([self.x - joint.x, self.y - joint.y, self.z - joint.z])

    def __add__(self, joint):
        return np.array([self.x + joint.x, self.y + joint.y, self.z + joint.z])
   
    def is_empty(self):
        return self.x == self.y == self.z == -1

    def get3DPoint(self):
        return np.array([self.x,self.y,self.z])
    
    def clone(self):
        joint = Joint()
        joint.id = self.id 
        joint.score = self.score  
        joint.x = self.x  
        joint.y = self.y  
        joint.z = self.z
        return joint
        
    def __str__(self):
        return "\n   Score={}, ID={}, point = ({:.2f},{:.2f},{:.2f})".format(self.score, self.id , self.x,self.y,self.z)




class Skeleton: 
    def __init__(self, obj = None):
        self.joints = {}
        self.id = 0
        self.score = 0

        for id in range(1,22): self.joints[id] = Joint()
        if obj is not None:
            self.id = obj.id
            self.score = obj.score
            ids = [kp.id for kp in obj.keypoints]
            for id in range(1,21):
                if id in ids:                
                    kp = obj.keypoints[ids.index(id)]
                    self.joints[id] = Joint(kp)
        self.create_root()
        self.root = self.joints[JOINTS["TORSO"]]

    
        self._ux = np.array([1, 0, 0])
        self._uy = np.array([0, 1, 0])
        self._uz = np.array([0, 0, 1])


    def clone(self):
        skl = Skeleton()
        skl.id = self.id
        skl.score=self.score
        skl.root = self.root
        for id, joint in self.joints.items():
            j = joint.clone()
            skl.joints[id]=j
        return skl


    def create_root(self):
        torso = Joint()
        center_hip = (self.joints[10] + self.joints[13])/2.
        center = (self.joints[3].get3DPoint() + center_hip)/2.
        torso.set_3DPoint(center)
        self.joints[21] = torso
    
    def GetJoint(self, id):
        return self.joints[id] if id in self.joints.keys() else Joint()
            


    def get_representation(self):
        self._update_axes()
        inclinations = []
        azimuths = []
        for key_i, key_j, key_k in CONNECTED_JOINTS:
            id_i, id_j, id_k= JOINTS[key_i], JOINTS[key_j], JOINTS[key_k]
            joint_i, joint_j, joint_k= self.joints[id_i], self.joints[id_j], self.joints[id_k]
            if joint_i.is_empty() or joint_j.is_empty() or joint_k.is_empty(): 
                inclinations.append(-1)
                azimuths.append(-1)
                continue

            pji = joint_i - joint_j
            pkj = joint_k - joint_j
            inclination = self._vecs_angle(pji, pkj)

            a = pji / (np.dot(pkj, pji) / np.dot(pji, pji))
            v1 = self._ux - a * np.dot(self._ux, pji)
            v2 = pkj - a * np.dot(pkj, pji)
            azimuth = self._vecs_angle(v1, v2)

            inclinations.append(inclination)
            azimuths.append(azimuth)

        azimuth = np.array(azimuths)
        inclination = np.array(inclinations)
        
        bending = np.array([ self.bending_angle(joint) for joint in self.joints.values()])
        rw, lw = self.joints[JOINTS['RIGHT_WRIST']], self.joints[JOINTS['LEFT_WRIST']]
        distances = [ self.pair_distances(rw,self.joints[key]) for key in JOINTS_DIST.values()]
        distances += [ self.pair_distances(lw,self.joints[key]) for key in JOINTS_DIST.values()]
        distances = np.array(distances)
        return np.concatenate([inclination,azimuth,bending,distances])

    def pair_distances(self, joint1,joint2):
        if joint1.is_empty() or joint2.is_empty(): return -1
        return np.linalg.norm(joint1 - joint2)

    def bending_angle(self,joint):
        if joint.is_empty():return -1
        return np.arccos(np.dot(self._uz, joint.get3DPoint()) / np.linalg.norm(joint.get3DPoint()))

    def _vecs_angle(self, v1, v2):
        return np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

    def _update_axes(self):
        self._ux = self.joints[JOINTS['RIGHT_SHOULDER']] - \
                   self.joints[JOINTS['LEFT_SHOULDER']]

        self._uy = self.joints[JOINTS['NECK']] - \
                   self.joints[JOINTS['TORSO']]

        self._ux = self._ux / np.linalg.norm(self._ux)
        self._uy = self._uy / np.linalg.norm(self._uy)

        proj_x_y = self._uy * np.dot(self._ux, self._uy)
        self._ux = self._ux - proj_x_y
        self._ux = self._ux / np.linalg.norm(self._ux)
        self._uz = np.cross(self._ux, self._uy)

    def vectorized(self):
        vec = np.ones((54))*(-1)
        for i in range(2,20):
            for key in self.joints.keys():
                j = self.joints[key]
                if j.id == i:
                    s = (i-2)*3
                    vec[s:s+3] = [j.x,j.y,j.z]
        return vec
    
    #It will normalize all links to a unit vector
    #Note that even small links as (head,eyes) will heve norm 1
    def normalize(self):
        skl = Skeleton()
        skl.id= self.id
        skl.score = self.score
        key_root = self.root.id
        skl.joints[key_root] = self.joints[key_root].clone()
        
        for start,end in LINKS:
            key_end = JOINTS[end]
            
            key_start = JOINTS[start]
            joint = Joint()

            joint.id =  self.joints[key_end].id
            joint.score =  self.joints[key_end].score
            
            joint_start, joint_end = self.joints[key_start], self.joints[key_end]
            if  joint_end.is_empty(): continue

            dist = (joint_start - joint_end)
            norm = np.sqrt((dist**2).sum()) + 1e-18
            dist = dist/norm
            end_points = skl.joints[key_start].get3DPoint() - dist
            joint.set_3DPoint(end_points)
            skl.joints[key_end] = joint
        return skl

    #Eliminate the lower joints and consider only uper limbs
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
    
#--------------- Functions for Data aumentation -------------------

    #Add a imall noise to the skeleton
    def add_noise(self):
        skl = self.clone()
        for joint in skl.joints.values():
            noise = np.random.normal(0, 0.01, 3)
            joint.set_3DPoint(joint.get3DPoint()+noise)
        return skl

    #Ramdomly erase a joint, which can simulate problems with the detector
    def erase_joint(self):
        joint_id = int(np.random.rand()*20)+1
        skl = self.clone()
        skl.joints[joint_id].set_3DPoint(np.zeros(3)-1)
        return skl

    #Flip the position of joints changing between left and right. 
    # It is for gesture performed with one hand. 
    # Note that it will not change the visulization of the gesture only the position
    def flip(self):
        flip_keys = { 1:1, 2:2, 3:3, 7:4, 
                      8:5, 9:6, 4:7, 5:8, 6:9,
                      13:10,14:11,15:12,10:13,
                      11:14,12:15,17:16,16:17,
                      19:18,18:19,20:20, 21:21
        }

        
        skl = Skeleton()
        skl.id= self.id
        skl.score = self.score
        skl.root= self.root.clone()
        for key in self.joints.keys():
            skl.joints[flip_keys[key]] = self.joints[key]
        skl.create_root()
        return skl
    #-------------------------------------------------------------
  
    
    #Normalize the representation to mean zero and variance 1
    def get_normalized_representation(self):
        representation = self.get_representation()
        indices = representation>-1
        values = representation[indices]
        mean = values.mean()
        std = values.std()+1e-18
        representation[indices] = (representation[indices]-mean)/std
        return representation
    
    #Put the bases of the joint to be in the root (TORSO)
    def centralize(self):
        skl = self.clone()
        for key,joint in skl.joints.items():
            if joint.is_empty():continue
            joint.set_3DPoint(joint - self.root)
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




def save_skl_vectors(files, file_name = "ufes_dataset"):
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
    np.save(file_name,np.array(dataset+dataset_flip))7


def save_skeletons(file_name = "ifes_dataset_skl"):
    files = glob.glob("/public/datasets/ifes-2018-10-19/*3d.json") 
    print(len(files))
    skl_dataset = []
    skl_labels = []
    dataset_flip = []
    for src in files:
        name = src.split("/")[-1].split("_")[0]
        #spots = glob.glob("/public/datasets/ufes-2020-01-23/spots_left_and_right_hand/{}_spots.json".format(name))[0]
        spots = glob.glob("/public/datasets/ifes-2018-10-19/{}_spots.json".format(name))[0]
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
                skl.label = labels[i]
                skl_dataset.append(skl)
                skl_labels.append(labels[i])
                break
    pickle.dump({"skeletons":skl_dataset, "labels":skl_labels}, open("{}.pkl".format(file_name),"wb"))
       

   
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
      
    ax_value = 4.0
    ax.clear()
    ax.set_xlim(-ax_value,ax_value)
    ax.set_ylim(-ax_value,ax_value)
    ax.set_zlim(-ax_value,ax_value)
    points = np.array([joint.get3DPoint() for joint in skeleton.joints.values() if not joint.is_empty()])
    ax.scatter(points[:,0], points[:,1], points[:,2])
    for s,e in LINKS:
        start = skeleton.joints[JOINTS[s]]
        end =  skeleton.joints[JOINTS[e]]
        if start.is_empty() or end.is_empty():continue
        points = np.array([start.get3DPoint(),end.get3DPoint()]) 
        ax.plot(points[:,0], points[:,1], points[:,2])
    
    torso  = skeleton.joints[1].get3DPoint()
    
    #neck  = skeleton.joints[3].get3DPoint()
    #ax.scatter([torso[0]], [torso[1]], [torso[2]],color="r")

    #ax.scatter([torso[0],neck[0]], [torso[1],neck[1]], [torso[2],neck[2]],color="r")
    
    fig.canvas.draw()
    plt.pause(0.0001)
    


if __name__ == "__main__":

    save_skeletons()
    # plt.ion()
    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    
    # data = pickle.load(open("ufes_dataset_skl.pkl","rb"))
    # for skl in data["skeletons"]:
    #             #skl.get_representation()
    #             plot_skeleton(ax, fig, skl.centralize().normalize().flip())




#     # In this case the skeletons are reading fram a JSON file. But it can be read from the IS.
#     json_files = glob.glob("./ifes_samples/p018g01_3d.json")
#     for src in json_files:
#         print(src)
#         with open(src) as f:
#             data = json.load(f)
#         for i, localization in enumerate(data["localizations"]):
            
#             annotations = ObjectAnnotations(localization)  
#             skeletons = [Skeleton(obj) for obj in annotations.objects]
                   
#             #plot all skeletons
#             for skl in skeletons:
#                 #skl.get_representation()
#                 plot_skeleton(ax, fig, skl.clone().centralize().normalize().erase_joint())  
#            


    
    
  











