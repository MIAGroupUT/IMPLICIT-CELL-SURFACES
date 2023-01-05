#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 31 21:43:24 2022

@author: xwiesne1
"""

from PIL import Image, ImageDraw
import numpy as np
from scipy import ndimage # distance transform
import matplotlib.pyplot as plt # plt.show()
import math

#from shape_generator_func import gen_shape

# rotate coordinates
# (point_x, point_y, origin_x, origin_y, angle)
def rotate_coord(ax, ay, bx, by, angle):
    # euclidean distance of a given point and a rotation origin
    radius = np.linalg.norm(np.array([ax,ay])-np.array([bx,by]))
    angle += math.atan2(ay-by, ax-bx)
    return (round(bx + radius * math.cos(angle)),
            round(by + radius * math.sin(angle)))

angle = 45

# triangle
# --------
points = ((128,56), (64,166), (192,166))
im = Image.new('L', (256,256))
draw = ImageDraw.Draw(im)
draw.polygon([rotate_coord(x, y, 128, 128,
                           math.radians(angle)) for x,y in points],
             fill='white')
#im.save('triangle.png')
#im.show()
trg_mask = np.asarray(im, dtype='bool')

# square
# ------
points = ((192,192), (192,64), (64,64), (64,192))
im = Image.new('L', (256,256))
draw = ImageDraw.Draw(im)
draw.polygon([rotate_coord(x, y, 128, 128,
                           math.radians(angle)) for x,y in points],
             fill='white')
#im.save('square.png')
#im.show()
sqr_mask = np.asarray(im, dtype='bool')

# compute sdf #################################################################

# triangle
# --------

# distance transform
edt_in = ndimage.distance_transform_edt(trg_mask)
edt_out = ndimage.distance_transform_edt(~trg_mask)

# clamp
edt_in[edt_in > 64] = 64.0
edt_out[edt_out > 64] = 64.0

# combine
sdf = edt_out - edt_in

# normalize to [-1,1]
sdf_norm = sdf / np.max(np.abs(sdf))

# check
#np.sum(trg_mask)
#np.sum(sdf_norm <= 0)

# visualize
plt.matshow(sdf_norm <= 0);
plt.colorbar()
plt.show()



## create function

age = 120

if age > 90:
    print("You are too old to party, granny.")
elif age < 0:
    print("You're yet to be born")
elif age >= 18:
    print("You are allowed to party")
else: 
    "You're too young to party"



bike = 'Yamaha'
 
if bike == 'Hero':
    print("bike is Hero")
 
elif bike == "Suzuki":
    print("bike is Suzuki")
 
elif bike == "Yamaha":
    print("bike is Yamaha")
 
else:
    print("Please choose correct answer")




