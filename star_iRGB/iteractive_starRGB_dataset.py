import numpy as np
from  glob import glob
from generate_iteractive_starRGB import *
import skimage as sk
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
class IterStarRGBDataset: 
    def __init__(self, dataset = "train", max_size = 24, alpha = 0.7, window = 3,transform = None):
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
        images,label = starRGB.get_images_only(self.files[index])
        sample = {"images":images, "label":torch.tensor([label])}
        if self.transform is not None:
            return self.transform(sample)
        return sample

    def __iter__(self):
        for idx in range(len(self.files)): yield self[idx]



class DataTrasformation(object):
    def __init__(self, output_size, data_aug = True):
        self.output_size = output_size
        self.data_aug = data_aug
        
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
            if np.random.rand(1) < 0.5:
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

        else:
            image = self.crop_center(image, self.output_size)
        
        
        image = np.transpose(image, (0,3,1,2)).astype(np.float32)
        image = torch.from_numpy(image.astype(np.float32))/255.0
        sample["images"] = image
        return sample


def create_datasets(num_workers = 2, batch_size = 32, max_size = 32, alpha = 0.7, window = 5, seq = 64):

    image_datasets = {
        "train":IterStarRGBDataset(dataset="train", max_size = max_size, alpha = alpha, window = window, 
                            transform=DataTrasformation(output_size=(110,120), data_aug = True)),
        "test":IterStarRGBDataset(dataset="test", max_size = max_size, alpha = alpha, window = window, 
                            transform=DataTrasformation(output_size=(110,120), data_aug = False)),
        "val":IterStarRGBDataset(dataset="validation", max_size = max_size, alpha = alpha, window = window,
                            transform=DataTrasformation(output_size=(110,120), data_aug = False))
                }

    dataloaders = {
        "train":DataLoader(image_datasets["train"], 
                                            batch_size=batch_size, pin_memory=True, shuffle=True, 
                                            num_workers=num_workers, drop_last=True),

        "test":DataLoader(image_datasets["test"], 
                                batch_size=1, pin_memory=True, shuffle=False, 
                                num_workers=num_workers, drop_last=False),

        "val":DataLoader(image_datasets["val"], 
                                batch_size=48, pin_memory=True, shuffle=False, 
                                  num_workers=num_workers, drop_last=True),
                 }
    return dataloaders  




if __name__ == "__main__":
    #params = torch.load("dynamic_star_rgb_9527.pth")["params"]
    #print(params)
    star = DynamicStarRGB(window = 5, channels=3, max_size = None, alpha = 0.6)
    directories = ["train", "test", "validation"]
    total = [0, 0, 0]

    for i,dir in enumerate(directories):
        samples = glob("/notebooks/datasets/Montalbano/{}/samples/*_color.mp4".format(dir))
        for sample in samples:
            name = sample.split("/")[-1].split("_")[0]
            images, labels = star.get_complete_images(sample)
            name = "/notebooks/datasets/Montalbano/{}/numpy_files/star_rgb_{}_{}".format( dir, name,len(labels))
            np.savez(name, images=images, labels=labels)
            total[i] += len(labels)
            print("{}  - {}/{}".format(name,len(images), len(labels)))
            
    print(total)
    for dir, t in zip(directories,total):
        print("Total {} = {}".format(dir,t))


