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
    An np.arry of the same size with the brightfield. Returns a float_type
    
    Notes
    -----------
    Casts image to an np.int32, which doesn't scale correctly, but is needed in 
    order to get negative values and clip them
    
    
    """
    
    #this allows for negative values
    image = img_as_float(image)
    
    three_d = len(image.shape) == 4
    
    new_image = np.copy(image)

    for channel in range(image.shape[-1]):
        if channel == bright:
            continue

        for sub_channel in range(image.shape[-1]):
            if (sub_channel == bright) or (channel == sub_channel):
                continue
            
            if three_d:
                new_image[:, :, :, channel] = new_image[:, :, :, channel] - image[:, :, :, sub_channel]
            else:
                new_image[:, :, channel] = new_image[:, :, channel] - image[:, :, sub_channel]
        

        
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
        return image
    
    elif (len(image.shape) == 3) and (image.shape[0] > 100):
        print("Looks like the image is not a z stack")
    
    
    type_proj = project_type.lower()
    type_lookup = {"max": np.max, "min": np.min, "mean": np.mean,
                   "median": np.median, "std": np.std, "sum": np.sum}
    func = type_lookup[type_proj]
    
    proj_image = func(image, axis=0)
    
    return proj_image
