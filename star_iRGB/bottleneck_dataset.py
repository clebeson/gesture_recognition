from generate_dynamic_star import *
import skimage as sk
from skimage import io, transform
from torchvision import transforms, models
from tqdm import tqdm
import threading
import torch

class BottleneckDataset: 
    def __init__(self, dataset = "train",sequence = None):
        self.seq = sequence
        files = glob.glob("/notebooks/datasets/Montalbano/{}/bottleneck/*.npz".format(dataset))
        self.samples = []
        self.indexes = []
        total = 0
        # ones = 0
        for idx, file in enumerate(files):
            sample = np.load(file)
            self.samples.append((sample["images"],sample["labels"]))
            size = len(sample["labels"])
            # ones += sum(sample["labels"]>0)
            if self.seq is None:
                b = 0
                e = size
                self.indexes.append((idx,b,e))
            else:
                for i in range(size//self.seq):
                    b = i*self.seq
                    e = b + self.seq
                    self.indexes.append((idx,b,e))
            total += size
        # print("umbalancing {}".format(ones/total))
        print("{} -> Samples {} |  Sequences {} | Total images {} |".format(dataset,len(files),len(self.indexes),total))

    def get_size(self,file):
        return int(file.split("_")[-1].replace(".npz",""))

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self,index):
        idx,b,e = self.indexes[index]
        images,labels = self.samples[idx]
        labels = labels[b:e]
        images = images[b:e]
        labels = torch.from_numpy(np.where(labels>0,1,0)).long()
        images = torch.from_numpy(images).float()
        return {"images":images, "label":labels}

    def __iter__(self):
        for images,labels in self.samples:
            labels = torch.from_numpy(np.where(labels>0,1,0)).long().unsqueeze(0)
            images = torch.from_numpy(images).float().unsqueeze(0)
            yield {"images":images, "label":labels}

    
