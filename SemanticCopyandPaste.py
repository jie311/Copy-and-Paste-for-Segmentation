import albumentations as A
import random
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt



class SemanticCopyandPaste():
    
    def __init__(self, nClass, path2rgb, path2mask):
        self.nClass     = nClass
        self.rgb_base   = path2rgb
        self.mask_base  = path2mask
        self.rgbs       = os.listdir(path2rgb)
        self.masks      = os.listdir(path2mask)
        self.nImages    = len(self.rgbs)
        self.threshold  = 30
        assert len(self.rgbs) == len(self.masks), "rgb path's file count != mask path's file count"
        
        
    
    def apply(self, image, mask):
        '''
            Input:
                image: 3-channel RGB images
                mask : 3-channel mask with pixel values of class index. Ex: 0 = background, 1 = class#1, etc

           This function will first randomly generate a class that being copied (Exclude 0, which is the background class). Then randomly picks a mask via provided path, and search whether it contains the previously picked target class. Keep randomly picks a new mask until a match is found. Finally start doing copy and paste process.

           Since semantic segmentation's annotation may not be labeled in the same way as instance segmentation therefore currently we copy and paste entire mask without further processing.

           TODO:
               1. Add zoom in and zoom out support before pasting (large scale jittering from paper)
               2. Mask shifting before pasting
               3.
        '''
        targetClass = 0
        while targetClass == 0:
            targetClass = random.randrange(self.nClass)
        
        print("Target Class = ", targetClass)
        
        ret = True
        
        while ret:
            candidate = random.randrange(self.nImages)
            c_image   = cv2.imread(os.path.join(self.rgb_base, self.rgbs[candidate]))   
            c_mask    = cv2.imread(os.path.join(self.mask_base, self.masks[candidate]))
            if self.target_class_in_image(c_mask, targetClass):
                ret = False
            
#             print("rgb =", c_image.shape)
#             print("mask =", c_mask.shape)
#             input()
        return self.copy_and_paste(c_image, c_mask, image, mask, self.nClass, targetClass)
    
    
    
    # Take 3-channel mask
    def target_class_in_image(self, mask, targetClassIdx):
        m   = mask[..., 0]
        tmp = (m==targetClassIdx)
        if np.sum(tmp) > self.threshold: #hard coded pixel threshold
            return True
        else:
            return False
    
    # Augmentation will be done on rgb2 (extract info from rgb1)
    # Extract class mask from rgb1 & mask1 and paste it on rgb2 & mask2
    def copy_and_paste(self, rgb1, mask1, rgb2, mask2, numClass, targetClassForAug):
        assert numClass > 0, "Incorrect class number"
        assert rgb1 is not None
        assert rgb2 is not None
        assert mask1 is not None
        assert mask2 is not None
        assert mask1.shape[2] == 3
        assert mask2.shape[2] == 3
        
        tmp = mask1[...,1]
        masks = [(tmp == v) for v in range(self.nClass)] # mask.shape = (x,y,ClassNums)
        masks = np.stack(masks, axis=-1).astype('float')
        
    #     assert np.sum(masks[...,targetClassForAug]) != 0, "Selected target class for augmentation has no content (all zeros)"

        
        newMask = mask2[...,1] - mask2[...,1] * masks[..., targetClassForAug] + targetClassForAug * masks[..., targetClassForAug] #new mask

        for i in range(3): #r,g,b channels
            rgb2[...,i] = rgb2[...,i] - rgb2[...,i]*masks[..., targetClassForAug] + rgb1[...,i] * masks[..., targetClassForAug]
            mask2[...,i]= newMask
        
        return rgb2, mask2


    __call__ = apply