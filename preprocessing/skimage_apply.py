#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Apply skimage function to every channel in an image



@author: pgb13
"""



def split_into_list(image, zstack=True):
    list_of_channels = []
    for channels in range(image.shape[-1]):
        if zstack is True:
            list_of_channels.append(image[:,:,:,channels])
        elif zstack is False:
            list_of_channels.append(image[:,:,channels])
        else:
            raise ValueError(f"zstack must be true or false, not {zstack}")
    return list_of_channels



def apply_to_channel(image_or_list, skimage_function, **kwargs):
    if type(image_or_list) != list:
        image_or_list = split_into_list(image_or_list)
    new_list = []
    for channel in image_or_list:
        new_list.append(skimage_function(channel, **kwargs))
    return new_list
