#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  2 22:27:02 2023

@author: xwiesne1
"""

from PIL import Image, ImageDraw
import numpy as np
from scipy import ndimage # distance transform
#import matplotlib.pyplot as plt # plt.show()
import math

# rotate coordinates
# (point_x, point_y, origin_x, origin_y, angle_rad)
def rotate_coord(ax, ay, bx, by, angle):
    # euclidean distance of a given point and a rotation origin
    radius = np.linalg.norm(np.array([ax,ay])-np.array([bx,by]))
    angle += math.atan2(ay-by, ax-bx)
    return (round(bx + radius * math.cos(angle)),
            round(by + radius * math.sin(angle)))

def gen_shape_sdf(shape_class, angle):
    if shape_class == 'square':
        points = ((192,192), (192,64), (64,64), (64,192))   
    elif shape_class == "triangle":
        points = ((128,56), (64,166), (192,166))   
    else:
        print("Incorect shape class.")
        exit()

    im = Image.new('L', (256,256))
    draw = ImageDraw.Draw(im)
    draw.polygon([rotate_coord(x, y, 128, 128,
                               math.radians(angle)) for x,y in points],
                 fill='white')
    shape_mask = np.asarray(im, dtype='bool')
    
    #return shape_mask
    
    # compute sdf
    # -----------
    
    # distance transform
    edt_in = ndimage.distance_transform_edt(shape_mask)
    edt_out = ndimage.distance_transform_edt(~shape_mask)

    # clamp
    edt_in[edt_in > 64] = 64.0
    edt_out[edt_out > 64] = 64.0

    # combine
    sdf = edt_out - edt_in

    # normalize to [-1,1]
    sdf_norm = sdf / np.max(np.abs(sdf))
    
    # check
    #np.sum(shape_mask)
    #np.sum(sdf_norm <= 0)
    
    return sdf_norm
