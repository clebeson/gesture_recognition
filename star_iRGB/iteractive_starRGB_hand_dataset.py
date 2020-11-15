import numpy as np
from  glob import glob
from generate_iteractive_starRGB import *
import skimage as sk
from torchvision import transforms
from torch.utils.data import DataLoader
import torch

class IterStarRGBHandDataset: 
    def __init__(self, dataset = "train", max_size = 24, alpha = 0.6, window = 5,transform = None, spotting = False):
        self.max_size = max_size
        self.channels=3
        self.transform = transform
        self.window = window
        self.alpha = alpha
        self.spotting = spotting
        if not spotting:
            self.files = glob("/notebooks/datasets/Montalbano/{}/videos/*.mp4".format(dataset))
            print("Dataset {}  with size ({},{},{},{},{})".format(dataset,len(self.files),self.max_size,120,160,self.channels))
        else:
            self.files = glob("/notebooks/datasets/Montalbano/{}/sample_images/images48_*.npz".format(dataset))
            print("Dataset {}  with size ({},{},{},{},{})".format(dataset,len(self.files),self.max_size,120,160,self.channels))
           
    def devide_datset(self):
        samples = glob("/notebooks/datasets/Montalbano/{}/sample_images/Sample*".format(dataset))
        self.indexes = []
        total = 0
        # ones = 0
        for sample in samples:
            frames = glob("{}/*.png".format(sample))
            size = len(frames)//2
            

            if self.max_size is None:
                b = 0
                e = size
                self.indexes.append([sample,b,e])
            else:
                for i in range(size//self.max_size):
                    b = i*self.max_size
                    e = b + self.max_size
                    self.indexes.append([sample,b,e])

        print("{} -> Samples {} |  Sequences {} | Total images {} |".format(dataset,len(samples),len(self.indexes),total))

        

    def __len__(self):
        return len(self.files) #if not self.spotting else len(self.indexes)


    def __getitem__(self,index):
        starRGB = DynamicStarRGB(self.window, self.channels, self.max_size, self.alpha, spotting = self.spotting)
        if not self.spotting:
            images,hands, label = starRGB.get_images(self.files[index])
            sample = {"images":images, "hands":hands,"label":torch.tensor([label])}
        else:
            # file, b,e = self.indexes[index]
            # images,hands, label = starRGB.get_chunk(file,b,e)
            # sample = {"images":images, "hands":hands,"label":torch.from_numpy(label)}
            data = np.load(self.files[index])
            sample = {"images":data["images"], "hands":data["hands"],"label":torch.from_numpy(data["label"])}


        if self.transform is not None:
            return self.transform(sample)
        return sample

    def __iter__(self):
        for index in range(len(self)):
            starRGB = DynamicStarRGB(self.window, self.channels, None, self.alpha, spotting = self.spotting)
            images, hands, label = starRGB.get_images(self.files[index])
            sample = {"images":images, "hands":hands,"label":torch.tensor([label]).long()}
            if self.transform is not None:
                yield self.transform(sample)
            else:
                yield sample
        # files = glob("/notebooks/datasets/Montalbano/test/sample_images/Sample*")
        # print(len(files))
        # for file in files:
        #     starRGB = DynamicStarRGB(self.window, self.channels, None, self.alpha, spotting = self.spotting)
        #     images,hands, label = starRGB.get_chunk_images(file)
        #     sample = {"images":images, "hands":hands,"label":torch.tensor([label]).long(), "file":file}
        #     if self.transform is not None:
        #         yield self.transform(sample)
        #     else: yield sample


class DataTrasformation(object):
    def __init__(self, output_size, data_aug = True):
        self.output_size = output_size
        self.data_aug = data_aug
        
    def randon_crop(self, images):
        image = images[0]
        height, width = self.output_size
        y = 0 if image.shape[1] - height <= 0 else np.random.randint(0, image.shape[1] - height)
        x = 0 if image.shape[2] - width <= 0 else np.random.randint(0, image.shape[2] - width)
        assert image.shape[2] >= width
        assert image.shape[1] >= height
        images[0] = image[:,y:y+height,x:x+width,:]
        return images

    def crop_center(self, images, out):
        image = images[0]
        y,x = image.shape[1], image.shape[2]
        cropy, cropx = out
        startx = x//2-(cropx//2)
        starty = y//2-(cropy//2) 
        startx = 0 if startx + cropx >=x  else startx
        starty = 0 if starty + cropy >=y else starty
        images[0] = image[:,starty:starty+cropy,startx:startx+cropx,:]
        return images

    def random_rotation(self,images):
            if np.random.rand(1) < 0.5:
                image = images[0]
                hand_left = images[1][0]
                hand_right = images[1][1]
                random_degree = np.random.uniform(-5, 5)
                image = np.array([sk.transform.rotate(image[i], random_degree) for i in range(len(image))])
                hand_left = np.array([sk.transform.rotate(hand_left[i], random_degree) for i in range(len(hand_left))])
                hand_left = np.array([sk.transform.rotate(hand_right[i], random_degree) for i in range(len(hand_right))])
                images[0] = image
                images[1][0] = hand_left
                images[1][1] = hand_right
            return images


    def random_noise(self,images):
        
        if np.random.rand(1) < 0.3:
            image = images[0]
            hand_left = images[1][0]
            hand_right = images[1][1]
            images[0] = sk.util.random_noise(image)
            images[1][0] = sk.util.random_noise(hand_left)
            images[1][1] = sk.util.random_noise(hand_right)

        return images
           

    def random_horizontal_flip(self,images):
        if np.random.rand(1) < 0.5:
            image = images[0]
            hand_left = images[1][0]
            hand_right = images[1][1]
            images[0] = image[:,:, ::-1,:]
            images[1][0] = hand_right[:,:, ::-1,:]
            images[1][1] = hand_left[:,:, ::-1,:]
        return images
           
   
    def __call__(self, sample):
        image = sample["images"]
        hands = sample["hands"]
        hands = [hands[:,:,:hands.shape[1]], hands[:,:,hands.shape[1]:]] #separete the right and left hands
        images = [image, hands]
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

        hands = np.concatenate(images[1],2)
        hands = np.transpose(hands, (0,3,1,2)).astype(np.float32)
        hands = torch.from_numpy(hands)/255.0
        sample["hands"] = hands

        return sample


def create_datasets(num_workers = 2, batch_size = 32, max_size = 32, alpha = 0.7, window = 5, spotting = False):

    image_datasets = {
        "train":IterStarRGBHandDataset(dataset="train", max_size = max_size, alpha = alpha, window = window, spotting = spotting,
                            transform=DataTrasformation(output_size=(110,120), data_aug = True)),
        "test":IterStarRGBHandDataset(dataset="test", max_size = max_size, alpha = alpha, window = window,  spotting = spotting,
                            transform=DataTrasformation(output_size=(110,120), data_aug = False)),
        "val":IterStarRGBHandDataset(dataset="validation", max_size = max_size, alpha = alpha, window = window, spotting = spotting,
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
                                batch_size=32, pin_memory=True, shuffle=False, 
                                  num_workers=num_workers, drop_last=True),
                 }
    return dataloaders  




if __name__ == "__main__":
    from tqdm import tqdm
    for name in ["train", "test", "validation"]:
        dataset = IterStarRGBHandDataset(name, max_size= 48, spotting = True)
        last_file = None
        for i in tqdm(range(len(dataset))):
            starRGB = DynamicStarRGB(max_size = None, spotting = True)
            file, b,e = dataset.indexes[i]
            if last_file != file:
                last_file = file
                images,hands, label = starRGB.get_chunk_images(file)
            sample = {"images":images[b:e], "hands":hands[b:e],"label":torch.from_numpy(label[b:e])}
            np.savez("/notebooks/datasets/Montalbano/{}/sample_images/images48_{}".format(name,i), **sample)



