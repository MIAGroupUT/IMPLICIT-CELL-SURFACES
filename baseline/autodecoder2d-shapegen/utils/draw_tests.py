#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 31 21:43:24 2022

@author: xwiesne1
"""

from PIL import Image, ImageDraw
import numpy as np
import math

# rotate coordinates
# (point_x, point_y, origin_x, origin_y, angle)
def rotate_coord(ax, ay, bx, by, angle):
    # euclidean distance of a point and a rotation origin
    radius = np.linalg.norm(np.array([ax,ay])-np.array([bx,by]))
    angle += math.atan2(ay-by, ax-bx)
    return (round(bx + radius * math.cos(angle)),
            round(by + radius * math.sin(angle)))

# triangle

#points = (128,72), (64, 182), (192,182)
points = (128,56), (64, 166), (192,166)

im = Image.new('L', (256, 256))
draw = ImageDraw.Draw(im)
#draw.polygon(points, fill='white') # outline='red', fill='blue'
draw.polygon([rotated_about(x,y, 128, 128, 
                             math.radians(angle)) for x,y in points],
               fill='white')
#im.save('triangle.png')
im.show()

triangle = np.asarray(im, dtype='bool')

# square
#(((Upper left x coordinate, upper left y coordinate),
# (lower right x coordinate, lower right y coordinate))
points = (64, 192, 192, 64)

im = Image.new('L', (256, 256))
draw = ImageDraw.Draw(im)
draw.rectangle(points, fill='white') # outline='red', fill='blue'
#im.save('square.png')
im.show()

#######################
# rotate

angle = 45

points = ((192,192),(192,64),(64,64),(64,192))

im = Image.new('L', (256, 256))
draw = ImageDraw.Draw(im)
#draw.rectangle(points, fill='white') 

draw.polygon([rotated_about(x,y, 128, 128, 
                             math.radians(angle)) for x,y in points],
               fill='white')
im.show()

#######################################
#finds the straight-line distance between two points
#def distance(ax, ay, bx, by):
#    return math.sqrt((by - ay)**2 + (bx - ax)**2)



image = Image.new('L', (100, 100), 127)
draw = ImageDraw.Draw(image)

square_center = (128,128)
square_length = 128

square_vertices = (
    (square_center[0] + square_length / 2, square_center[1] + square_length / 2),
    (square_center[0] + square_length / 2, square_center[1] - square_length / 2),
    (square_center[0] - square_length / 2, square_center[1] - square_length / 2),
    (square_center[0] - square_length / 2, square_center[1] + square_length / 2)
)

square_vertices = [rotated_about(x,y, square_center[0], square_center[1], 
                                 math.radians(45)) for x,y in square_vertices]

draw.polygon(square_vertices, fill=255)

image.show()

image.save("output.png")