
import numpy as np
import glob
import pickle
import torch
import skimage as sk
import scipy.misc as m
import random 
from skimage import io, transform
from torchvision import transforms, models


class HandTrasformation(object):
    def __init__(self, output_size, data_aug = True):
        self.output_size = output_size
        self.data_aug = data_aug
       
  
    def random_jitter(self, image):
        if np.random.rand(1) > 0.2:
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
            if np.random.rand(1) < 0.5:
                random_degree = np.random.uniform(-10, 10)
                image = np.array([sk.transform.rotate(image[i], random_degree) for i in range(len(image))])
            return image


    def random_noise(self,image_array,skls):
        if np.random.rand(1) < 0.5:
            noise = np.random.randn(*image_array.shape)*10
            skls = [skl.add_noise() for skl in skls]
            noisy_image = image_array+noise
            noisy_image = np.where(np.bitwise_or(image_array<=0, noisy_image < 0), 0, noisy_image)
            image_array = np.where(noisy_image > 255, 255, noisy_image)
        return image_array, skls
           

    def random_horizontal_flip(self,image_array,skls):
        if np.random.rand(1) < 0.5:
            skls = [skl.flip() for skl in skls]
            image_array = image_array.reshape(-1,4,2,image_array.shape[-3], image_array.shape[-2],image_array.shape[-1])
            lh = np.copy(image_array[:,:,0])
            ll = np.copy(image_array[:,:,1])
            image_array[:,:,0] = ll
            image_array[:,:,1] = lh
            image_array = image_array.reshape(-1,image_array.shape[-3], image_array.shape[-2],image_array.shape[-1])
            
        return image_array,skls
           
    def unsplit(self,images):
        h,w = self.output_size[0], self.output_size[1]
        images = images.reshape(-1,8, h,w,3)
        image_result = np.zeros(images.shape).reshape(-1, 2, 2*h,2*w,3)
        image_result[:,0,:h,:w] = images[:,0]
        image_result[:,0,:h,w:2*w] = images[:,1]
        image_result[:,0,h:,:w] = images[:,2]
        image_result[:,0,h:,w:2*w] = images[:,3]

        image_result[:,1,:h,:w] = images[:,4]
        image_result[:,1,:h,w:2*w] = images[:,5]
        image_result[:,1,h:,:w] = images[:,6]
        image_result[:,1,h:,w:2*w] = images[:,7]
        return image_result

    
    def __call__(self, sample):
        image,skls = sample["images"],sample['skeletons']
        if self.data_aug:
            # image = self.randon_crop(image)
            image,skls = self.random_horizontal_flip(image,skls)
            image = self.random_rotation(image)
            image,skls = self.random_noise(image,skls)
            # image = torch.from_numpy(image.astype(np.float32))
            # image = transforms.RandomErasing()(image)
            # image = self.random_jitter(image)
            # image = image.permute(0,1,5,2,3,4)/255.0
            # return {'image': image, 'label': torch.from_numpy(np.array(label)).long()}

        else:
            # image = sk.transform.resize(image,(120,140))
            image = self.crop_center(image, self.output_size)


        
        skls = torch.tensor([skl.centralize().normalize().vectorize_reduced() for skl in skls]).float()
        # print(image.shape, skls.shape)
        
        # image = image.reshape(-1,4,2, self.output_size[0],self.output_size[1],3)
        image = self.unsplit(image)
        # hand_l = image[1,:,0]
        # hand_l = image[1,:,0]
        # m.imsave('images/image.png', image[10])
        # print( "inside  ---> ",image.max(), image.min())
        # image = np.transpose(image, (0,2,5,1,3,4)).astype(np.float32)
        
        image = np.transpose(image, (0,1,4,2,3)).astype(np.float32)
        image = torch.from_numpy(image)/255.0
        # mean = image.mean()
        # std = image.std() + 1e-8
        # image = (image - mean)/std
        sample['images'] = image
        sample['skeletons'] = skls


        # # print(image.shape)
        return sample
    