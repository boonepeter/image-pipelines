#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preprocess images

@author: pgb13
"""




import numpy as np

from skimage import img_as_float




def subtract_channels(image, bright=None):
    """Subtracts every channel from each other in a multichannel image
    
    Parameters
    ----------
    image: np.array of any dtype
    bright: brightfield channel to ignore and not subtract. Default is none
    
    Return
    -----------
    An np.arry of the same size with the brightfield
    
    """
    
    
    image = img_as_float(image)
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
                
def z_project(image, project_type="max"):
    """Projects an image along the first axis using a chosen method.
    
    Parameters
    ----------
    image = np.array with the first axis = the z sections
    project_type = sting from list: ['max', 
                                     'min',
                                     'mean',
                                     'median',
                                     'std',
                                     'sum']
    
    Returns
    ----------
    flattened image
    
    """
    image = img_as_float(image)
    
    if (len(image.shape) < 3):
        print(f"Image has only {len(image.shape)} dimension(s)")
        return
    
    elif (len(image.shape) == 3) and (image.shape[0] > 100):
        print("Looks like the image is not a z stack")
    
    
    type_proj = project_type.lower()
    type_lookup = {"max": np.max, "min": np.min, "mean": np.mean,
                   "median": np.median, "std": np.std, "sum": np.sum}
    func = type_lookup[type_proj]
    
    proj_image = np.apply_along_axis(func1d=func, axis=0, arr=image)
    
    return proj_image
    

    
    
    
    
    



