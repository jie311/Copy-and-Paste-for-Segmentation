import albumentations as A
import random
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt



class SemanticCopyandPaste(A.DualTransform):
    
    def __init__(self, 
                 nClass, 
                 path2rgb, 
                 path2mask, 
                 shift_x_limit = None, 
                 shift_y_limit = None, 
                 always_apply=False, 
                 p=0.5):
        super().__init__(always_apply=always_apply, p=p)
        self.nClass     = nClass
        self.rgb_base   = path2rgb
        self.mask_base  = path2mask
        self.rgbs       = os.listdir(path2rgb)
        self.masks      = os.listdir(path2mask)
        self.nImages    = len(self.rgbs)
        self.threshold  = 30
        self.targetClass= 0
        self.c_image    = None # candidate image
        self.c_mask     = None # candidate mask
        self.found      = False
        self.imgRow     = None  # for image translation
        self.imgCol     = None  # for image translation
        self.shift_x_limit = shift_x_limit
        self.shift_y_limit = shift_y_limit
        self.translated_mask = None
        
        assert len(self.rgbs) == len(self.masks), "rgb path's file count != mask path's file count"
        assert self.nClass > 0, "Incorrect class number"
        if shift_x_limit is not None:
            assert type(shift_x_limit) == list and type(shift_y_limit) == list
        
    
    def apply(self, image, **params):
        '''
            Args:
                image: 3-channel RGB images

           This function will first randomly generate a class that being copied (Exclude 0, which is the background class). Then randomly picks a mask via provided path, and search whether it contains the previously picked target class. Keep randomly picks a new mask until a match is found. Finally start doing copy and paste process.

           Since semantic segmentation's annotation may not be labeled in the same way as instance segmentation therefore currently we copy and paste entire mask without further processing.

           TODO:
               1. Add zoom in and zoom out support before pasting (large scale jittering from paper)
               2. Mask shifting before pasting
               3.
        '''
        
        
        self.targetClass = random.randint(1,self.nClass-1) # not aug background class
        
        ret = True
        while ret:
            candidate = random.randrange(self.nImages)
            c_mask    = cv2.imread(os.path.join(self.mask_base, self.masks[candidate]))
            if self.target_class_in_image(c_mask, self.targetClass):
                c_image   = cv2.imread(os.path.join(self.rgb_base, self.rgbs[candidate]))
                c_image   = cv2.cvtColor(c_image, cv2.COLOR_BGR2RGB)
                ret              = False
                self.found       = True
                self.c_mask      = c_mask
                self.c_image     = c_image
                
        return self.copy_and_paste_image(self.c_image, self.c_mask, image, self.targetClass)
    
    
    
    def apply_to_mask(self, mask, **params):
        assert self.found == True
        return self.copy_and_paste_mask(self.c_mask, mask, self.targetClass)
    

    
    
    # Augmentation will be added to rgb2 (extract content from rgb1)
    # Mask1 is need to know where to extract pixels for color image copy and paste
    def copy_and_paste_image(self, rgb1, mask1, rgb2, targetClassForAug):
        assert rgb1 is not None
        assert rgb2 is not None
        assert mask1 is not None
        assert mask1.shape[2] == 3 # We imread it without further process, so its a 3 channel
        
        tmp   = mask1[...,1] # All 3 channels have same content, we take 1 to process
        masks = [(tmp == v) for v in range(self.nClass)] 
        masks = np.stack(masks, axis=-1).astype('float') # mask.shape = (x,y,ClassNums)
        self.c_mask = masks

        
        if self.shift_x_limit is not None: # That we are doing translation
            masks[..., targetClassForAug] = self.imgTranslate(masks[..., targetClassForAug], self.shift_x_limit, self.shift_y_limit)
            
        self.translated_mask = masks[..., targetClassForAug]
        # unroll for loop
        rgb2[...,0] = rgb2[...,0] - rgb2[...,0]*self.translated_mask + rgb1[...,0] * self.translated_mask
        rgb2[...,1] = rgb2[...,1] - rgb2[...,1]*self.translated_mask + rgb1[...,1] * self.translated_mask
        rgb2[...,2] = rgb2[...,2] - rgb2[...,2]*self.translated_mask + rgb1[...,2] * self.translated_mask
        
        
        return rgb2

    
    
    
    def copy_and_paste_mask(self, mask1, mask2, targetClassForAug):
        '''
            Args:
                mask1 = randomly picked qualified mask from apply(), has shape = (x, y, nClasses)
                mask2 = dataloader loaded mask, aug is added to mask2
        '''
        assert mask2.shape[2] == self.nClass # Processed by dataloader, so its a nClass channel
        assert self.translated_mask is not None
        
        mask2_1channel = np.argmax(mask2, axis=2)
        
        newMask = mask2_1channel - mask2_1channel * self.translated_mask + targetClassForAug * self.translated_mask
        
        
        masks = [(newMask == v) for v in range(self.nClass)] # mask.shape = (x,y,ClassNums)
        masks = np.stack(masks, axis=-1).astype('float')
        
        # Reset
        self.c_mask = None 
        self.found == False
        self.translation_matrix = None
        self.translated_mask = None
        return masks
    
    
    
    
    
    
    # We imread the mask, so it's a 3-channel mask (not one-hot encoded)
    def target_class_in_image(self, mask, targetClassIdx):
        m   = mask[..., 0]
        tmp = (m==targetClassIdx)
        if np.sum(tmp) > self.threshold: #hard coded pixel threshold
            return True
        else:
            return False
    
    
    def apply_to_bbox(self, bbox, **params):
        return bbox

    def apply_to_keypoint(self, keypoint, **params):
        return keypoint

    def get_transform_init_args_names(self):
        return ("image", "mask")
    

        
    def imgTranslate(self, image, offset_x_limit, offset_y_limit ):
        '''
            Args:
                image: it can be mask or rgb image
                offset_x_limt: x-axis shift limit [-1,1]
                offset_y_limt: y-axis shift limit [-1,1]
        '''
        if self.imgRow == None or self.imgCol == None:
            self.imgRow, self.imgCol = image.shape

        if self.translation_matrix is None:
            col_shift = random.uniform(offset_x_limit[0], offset_x_limit[1])*self.imgCol
            row_shift = random.uniform(offset_y_limit[0], offset_y_limit[1])*self.imgRow
            translation_matrix = np.float32([[1,0,col_shift], [0,1,row_shift]])
        
        shifted_img = cv2.warpAffine(image, translation_matrix, (self.imgCol, self.imgRow))

        return shifted_img

