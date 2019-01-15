#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preprocess images

@author: pgb13
"""




import numpy as np

from skimage.filters import gaussian
from skimage import img_as_float




def subtract_channels(image, gaus=False, bright=None):
    """Subtracts every channel from each other in a multichannel image
    
    Parameters
    ----------
    image: np.array of any dtype
    gaus: False or int. If int, gaussian sigma = int
    bright: brightfield channel to ignore and not subtract. Default is none
    
    Return
    -----------
    An np.arry of the same size with the brightfield
    
    """
    
    
    image = img_as_float(image)
    
    if gaus:
        image = gaussian(image)
        
    three_d = len(image.shape) == 4
    
    new_image = np.zeros(shape=image.shape, dtype=image.dtype)

    for channel in range(image.shape[-1]):
        if channel == bright:
            if three_d:
                new_image[:, :, :, channel] = image[:, :, :, channel]
            else:
                new_image[:, :, channel] = image[:, :, channel]
            
            continue
        
        if three_d:
            this_chan = np.copy(image[:, :, :, channel])
        else:
            this_chan = np.copy(image[:, :, channel])
        
        for sub_channel in range(image.shape[-1]):
            if (sub_channel == bright) or (channel == sub_channel):
                continue
            
            if three_d:
                this_chan = this_chan - image[:, :, :, sub_channel]
            else:
                this_chan = this_chan - image[:, :, sub_channel]
        
        if three_d:
            new_image[:, :, :, channel] = this_chan
        else:
            new_image[:, :, channel] = this_chan
        
        
    new_image = np.clip(new_image, a_min=image.min(), a_max=image.max())
    return new_image
                
        
    
    



