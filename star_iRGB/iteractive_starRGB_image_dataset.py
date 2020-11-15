import numpy as np
from  glob import glob
from generate_iteractive_starRGB import *
import skimage as sk
from torchvision import transforms
from torch.utils.data import DataLoader
import torch

class IterStarRGBContextDataset: 
    def __init__(self, dataset = "train", max_size = 24, alpha = 0.6, window = 5,transform = None):
        self.max_size = max_size
        self.channels=3
        self.transform = transform
        self.window = window
        self.alpha = alpha
        self.files = glob("/notebooks/datasets/Montalbano/{}/videos/*.mp4".format(dataset))
        print("Dataset {}  with size ({},{},{},{},{})".format(dataset,len(self.files),self.max_size,120,160,self.channels))
           
        

    def __len__(self):
        return len(self.files) 


    def __getitem__(self,index):
        starRGB = DynamicStarRGB(self.window, self.channels, self.max_size, self.alpha)
        images,context, label = starRGB.get_StarRGB_and_images(self.files[index])
        sample = {"images":images, "context":context,"label":torch.tensor([label])}
        if self.transform is not None:
            return self.transform(sample)
        return sample




class DataTrasformation(object):
    def __init__(self, output_size, data_aug = True):
        self.output_size = output_size
        self.data_aug = data_aug
        
    def randon_crop(self, images):
        image = images[0]
        context = images[1]
        height, width = self.output_size
        y = 0 if image.shape[1] - height <= 0 else np.random.randint(0, image.shape[1] - height)
        x = 0 if image.shape[2] - width <= 0 else np.random.randint(0, image.shape[2] - width)
        assert image.shape[2] >= width
        assert image.shape[1] >= height
        images[0] = image[:,y:y+height,x:x+width,:]
        images[1] = context[:,y:y+height,x:x+width,:]

        return images

    def crop_center(self, images, out):
        image = images[0]
        context = images[1]
        y,x = image.shape[1], image.shape[2]
        cropy, cropx = out
        startx = x//2-(cropx//2)
        starty = y//2-(cropy//2) 
        startx = 0 if startx + cropx >=x  else startx
        starty = 0 if starty + cropy >=y else starty
        images[0] = image[:,starty:starty+cropy,startx:startx+cropx,:]
        images[1] = context[:,starty:starty+cropy,startx:startx+cropx,:]
        return images

    def random_rotation(self,images):
            if np.random.rand(1) < 0.5:
                image = images[0]
                context = images[1]
                random_degree = np.random.uniform(-5, 5)
                images[0] = np.array([sk.transform.rotate(image[i], random_degree) for i in range(len(image))])
                images[1] = np.array([sk.transform.rotate(context[i], random_degree) for i in range(len(context))])
            return images


    def random_noise(self,images):
        
        if np.random.rand(1) < 0.3:
            image = images[0]
            context = images[1]
            images[0] = sk.util.random_noise(image)
            images[1] = sk.util.random_noise(context)
        return images
           

    def random_horizontal_flip(self,images):
        if np.random.rand(1) < 0.5:
            image = images[0]
            context = images[1]
            images[0] = image[:,:, ::-1,:]
            images[1] = context[:,:, ::-1,:]
        return images
           
   
    def __call__(self, sample):
        images = [sample["images"], sample["context"]]
        if self.data_aug:
            images = self.randon_crop(images)
            images = self.random_horizontal_flip(images)
            images = self.random_rotation(images)
            images = self.random_noise(images)
        else:
            images = self.crop_center(images, self.output_size)
        
        image = images[0]
        image = np.transpose(image, (0,3,1,2)).astype(np.float32)
        image = torch.from_numpy(image)/255.0
        sample["images"] = image

        context = images[1]
        context = np.transpose(context, (0,3,1,2)).astype(np.float32)
        context = torch.from_numpy(context)/255.0
        sample["context"] = context

        return sample


def create_datasets(num_workers = 2, batch_size = 32, max_size = 32, alpha = 0.7, window = 5):

    image_datasets = {
        "train":IterStarRGBContextDataset(dataset="train", max_size = max_size, alpha = alpha, window = window, 
                            transform=DataTrasformation(output_size=(110,120), data_aug = True)),
        "test":IterStarRGBContextDataset(dataset="test", max_size = max_size, alpha = alpha, window = window,  
                            transform=DataTrasformation(output_size=(110,120), data_aug = False)),
        "val":IterStarRGBContextDataset(dataset="validation", max_size = max_size, alpha = alpha, window = window, 
                            transform=DataTrasformation(output_size=(110,120), data_aug = False))
                }

    dataloaders = {
        "train":DataLoader(image_datasets["train"], 
                                            batch_size=batch_size, pin_memory=True, shuffle=True, 
                                            num_workers=num_workers, drop_last=True),

        "test":DataLoader(image_datasets["test"], 
                                batch_size=30, pin_memory=True, shuffle=False, 
                                num_workers=num_workers, drop_last=False),

        "val":DataLoader(image_datasets["val"], 
                                batch_size=48, pin_memory=True, shuffle=False, 
                                  num_workers=num_workers, drop_last=True),
                 }
    return dataloaders  




if __name__ == "__main__":
    from tqdm import tqdm
    for name in ["train", "test", "validation"]:
        dataset = IterStarRGBContextDataset(name)
        for i in tqdm(range(len(dataset))):
            sample = dataset[i]
            np.savez("/notebooks/datasets/Montalbano/{}/sample_images/images_{}".format(name,i), **sample)



