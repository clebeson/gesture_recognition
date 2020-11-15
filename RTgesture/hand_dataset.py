
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


class HandTrasformation(object):
    def __init__(self, output_size, data_aug = True):
        self.output_size = output_size
        self.data_aug = data_aug
        
    def random_jitter(self, image):
        if np.random.rand(1) > 0.3:
            image = transforms.ColorJitter(*(np.random.rand(4)*0.3) )(image)
        return image

    def randon_crop(self, image):
        height, width = self.output_size
        y = 0 if image.shape[1] - height <= 0 else np.random.randint(0, image.shape[1] - height)
        x = 0 if image.shape[2] - width <= 0 else np.random.randint(0, image.shape[2] - width)
        assert image.shape[2] >= width
        assert image.shape[1] >= height
        return image[:,y:y+height,x:x+width,:]

    def crop_center(self, image, out):
        y,x = image.shape[1], image.shape[2]
        cropy, cropx = out
        startx = x//2-(cropx//2)
        starty = y//2-(cropy//2) 
        startx = 0 if startx + cropx >=x  else startx
        starty = 0 if starty + cropy >=y else starty

        return image[:,starty:starty+cropy,startx:startx+cropx,:]

    def random_rotation(self,image):
          # pick a random degree of rotation between 25% on the left and 25% on the right
            if np.random.rand(1) < 0.3:
                random_degree = np.random.uniform(-5, 5)
                image = np.array([sk.transform.rotate(image[i], random_degree) for i in range(len(image))])
            return image


    def random_noise(self,image_array):
        if np.random.rand(1) < 0.3:
            image_array = sk.util.random_noise(image_array)
        return image_array
           

    def random_horizontal_flip(self,image_array):
        if np.random.rand(1) < 0.5:
            image_array = image_array[:,:, ::-1,:]
        return image_array
           
   
    def __call__(self, sample):
        image = sample["images"]
        if self.data_aug:
            image = self.randon_crop(image)
            image = self.random_horizontal_flip(image)
            image = self.random_rotation(image)
            image = self.random_noise(image)
            # image = torch.from_numpy(image.astype(np.float32))
            # image = transforms.RandomErasing()(image)
            # image = self.random_jitter(image)
            # image = image.permute(0,1,5,2,3,4)/255.0
            # return {'image': image, 'label': torch.from_numpy(np.array(label)).long()}

        else:
            # image = sk.transform.resize(image,(120,140))
            image = self.crop_center(image, self.output_size)
        
        
        image = image.reshape(-1,2,4,self.output_size[0],self.output_size[1],3)
        image = np.transpose(image, (0,1,5,2,3,4)).astype(np.float32)
        image = torch.from_numpy(image.astype(np.float32))/255.0
        sample["images"] = image


        # mean = image.view(image.size(0), -1).mean(1)
        # std = image.view(image.size(0), -1).std(1)+ 1e-18
        # image = (image - mean.view(-1,1,1))/std.view(-1,1,1)
        # # print(image.shape)
        return sample


class HandDataset(Dataset):
    """Flower dataset."""

    def __init__(self, type = "train", max_seq = None, transform=None):
        self.type = type
        self.max_seq = max_seq
        self.transform = transform

        dataset_name = "ifes" if type== "train" else "ufes"

        self.files = glob.glob("hand_images/{}*.jpg".format(dataset_name))
        self.files = sorted(self.files,key = lambda  name:int(name.replace(".jpg","").split("_")[-1]))
        # persons = np.array([int(name.split("_")[-3]) for name in self.files])
        # if type == "train":
        #     self.files = [self.files[i] for i, p in enumerate(persons) if p not in [17,18]]
        # else:
        #     self.files = [self.files[i] for i, p in enumerate(persons) if p in [17,18]]
        
        self.labels = np.array([int(name.split("_")[-2]) for name in self.files])
        
        self.videos = self._segment_videos(self.labels)
       
        print("{} - files = {}, labels = {}, videos = {}, classes = {}".format(type,len(self.files),len(self.labels),len(self.videos),len(np.unique(self.labels))   )  )
        # if type == "train": self.videos, self.labels = self.load_data()
    
    def _segment_videos(self, labels):
        begin = 0
        prev = int(labels[0])
        videos = []
               
        for i,label in enumerate(labels):
            if label != prev: 
                if prev == 0 : begin = i
                else: 
                    videos.append([begin,i])
                    begin = 0
                prev = label

        if begin != 0:videos.append([begin,i])

        return np.array(videos).astype(int)


    def __len__(self):
        return len(self.videos)

    def get_data(self):
        print("loading {} ...".format(self.type))
        images = []
        labels = []
        for begin,end in self.videos:
            labels.append(self.labels[begin:end])
            video_hands = np.array([self._split_image(io.imread(img_name)) for img_name in self.files[begin:end]])
            video_hands = video_hands.reshape((-1,60,60,3))
            # video_hands = np.transpose(video_hands, (0,2,5,1,3,4))
            if self.transform: video_hands = self.transform({'images': video_hands})["images"]
            images.append(video_hands)
            
        return images, labels
    
    
    def load_data(self):
        print("loading {} ...".format(self.type))
        hands = []
        labels = []
        for begin,end in self.videos[:10]:
            images =  np.zeros((self.max_seq,8,60,60,3))
            label = torch.tensor([self.labels[begin]]*self.max_seq).long()-1
            labels.append(label)
            for i,img_name in zip(range(self.max_seq),self.files[begin:end]):
                images[i] = self._split_image(io.imread(img_name))
            images = images.reshape((-1,60,60,3))
            # images = np.transpose(images, (0,2,5,1,3,4))
            hands.append(images)
        print("Loaded {} images with shape {}".format(len(images),images[0].shape))
        return hands, labels
            
           

    def _split_image(self,image):
        if len(image.shape) == 3:
            image = np.expand_dims(image,0)
        image = np.array([[image[:,:60,i*60:(i+1)*60],image[:,60:,i*60:(i+1)*60]] for i in range(4)])
        return image.reshape((-1,60,60,3))


    def __getitem__(self, idx):
        begin, end = self.videos[idx]
        files = self.files[begin:end]
        images =  np.zeros((self.max_seq,120,240,3))
        labels = torch.tensor([self.labels[begin]]*self.max_seq).long()-1
        for i,img_name in zip(range(self.max_seq),files):
            images[i] = io.imread(img_name)
        
        images = self._split_image(images)

        sample = {'images': images, 'labels': labels}
        if self.transform:
            sample = self.transform(sample)
        else:
            sample['images'] =  torch.from_numpy(sample['images']).float()/255.0, 
        return sample    