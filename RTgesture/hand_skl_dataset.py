
import numpy as np
import glob
import pickle
import torch
import skimage as sk
import random 
from skimage import io, transform
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from skeleton import Skeleton


class HandSKLDataset(Dataset):
    """Flower dataset."""

    def __init__(self, type = "train", max_seq = None, transform=None, spotting = False):
        self.type = type
        self.max_seq = max_seq
        self.transform = transform
        self.spotting = spotting


        dataset_name = "ifes" if type== "train" else "ufes"
        self.skl = pickle.load(open("{}_{}dataset_skl.pkl".format(dataset_name, "complete_" if dataset_name == "ufes" else ""),"rb"))
        self.labels = self.skl["labels"]
        self.skl = self.skl["skeletons"]

        # self.test_skl = pickle.load(open("ufes_dataset_skl.pkl","rb"))
        # self.test_skl_labels = self.test_skl["labels"]
        # self.test_skl = self.test["skeletons"]

        self.skl_enconded = False

        self.hand = glob.glob("hand_images/ifes*.jpg" if dataset_name == "ifes" else "images_ufes/ufes*.png" )
        self.hand = sorted(self.hand,key = lambda  name:int(name.replace(".jpg" if dataset_name == "ifes" else ".png","").split("_")[-1]))
        # persons = np.array([int(name.split("_")[-3]) for name in self.files])
        # if type == "train":
        #     self.files = [self.files[i] for i, p in enumerate(persons) if p not in [17,18]]
        # else:
        #     self.files = [self.files[i] for i, p in enumerate(persons) if p in [17,18]]
        # self.labels = np.array([float(name.split("_")[-2]) for name in self.files]).astype(int)
        
        self.videos = self._segment_videos(self.labels)
        img = io.imread(self.hand[0])
        # print(self.hand[0:10])
        
        print("{} - hand/skl = {}/{}, labels = {}, videos = {}, unique = {} hand_shape = {}  hand min/max = {}/{}".format(type,len(self.hand),len(self.skl),len(self.labels),len(self.videos),len(np.unique(self.labels)), img.shape,img.min(), img.max()   )  )
    
    
    def _segment_videos(self, labels):
        begin = 0
        prev = int(labels[0])
        videos = []

        if self.spotting:
        
            size = (len(labels)//self.max_seq) 
            for i in range(size):
                b = i*self.max_seq
                e = b+self.max_seq
                videos.append([b,e])
        else:
            for i,label in enumerate(labels):
                if label > 0 and prev == 0: 
                    begin = i
                elif prev > 0 and label == 0:
                    videos.append([begin,i])
                    begin = 0
                prev = label

            if label > 0:videos.append([begin,i])

        return np.array(videos).astype(int)

    def __len__(self):
        return len(self.videos)

    def get_data(self):
        print("loading {} ...".format(self.type))
        images = []
        labels = []
        skeletons = []
        sizes = []
        for idx in range(len(self.videos)):
            sample = self[idx]
            # print( "inside  ---> ",sample["images"].max(), sample["labels"].max(), sample["skeletons"].max())
            labels.append(sample["labels"])
            images.append(sample["images"])
            skeletons.append(sample["skeletons"])
            sizes.append(len(sample["labels"]))
        print("Loaded {} samples".format(len(images)))
        print(sum(sizes)/len(sizes), max(sizes), min(sizes))
        return images, skeletons, labels
            

    def _split_image(self,image):
        if len(image.shape) == 3:
            image = np.expand_dims(image,0)
        image = np.array([[image[:,:50,i*50:(i+1)*50],image[:,50:,i*50:(i+1)*50]] for i in range(4)])
        return image.reshape((-1,50,50,3))


    def __getitem__(self, idx):
        begin, end = self.videos[idx]
        files = self.hand[begin:end]
        skls = self.skl[begin:end]
        max_seq = self.max_seq if self.max_seq is not None else len(files)
        images =  np.zeros((max_seq,100,200,3))-1
        skeletons = []
        if self.spotting:
            labels = np.array(self.labels[begin:end])
            # labels = torch.from_numpy(np.where(labels > 0,1 , 0)).long()
        else:
            labels = torch.tensor([self.labels[begin]]*max_seq).long()-1
        for i,img_name,skl in zip(range(max_seq),files,skls):
            images[i] = transform.resize(io.imread(img_name), (100,200), preserve_range = True).astype(np.uint8)
            skeletons.append(skl)
        while len(skeletons) < max_seq:skeletons.append(Skeleton())
        images = self._split_image(images)

        sample = {'images': images, 'skeletons':skeletons, 'labels': labels}
        if self.transform:
            sample = self.transform(sample)
        else:
            sample['skeletons'] = torch.tensor([skl.centralize().normalize().get_normalized_representation() for skl in sample['skeletons'] ]).float()
            image = sample['images']
            image = image.reshape(-1,4,2,self.output_size[0],self.output_size[1],3)
            image = np.transpose(image, (0,2,5,1,3,4)).astype(np.float32)
            sample['images'] = torch.from_numpy(image)
            # sample['skeletons'] = torch.tensor([skl.centralize().normalize().vectorize_reduced() for skl in sample['skeletons']]).float()
            # sample['images'] =  torch.from_numpy(sample['images']).float()/255.0, 
        return sample    