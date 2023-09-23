#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  2 22:47:00 2023

@author: xwiesne1
"""

#import numpy as np
import matplotlib.pyplot as plt # plt.show()

from shape_generator_func import gen_shape_sdf

# generate shape
angle_deg = 180
#sdf = gen_shape_sdf(angle_deg, 'triangle')
sdf = gen_shape_sdf('square', angle_deg)

# visualize
plt.matshow(sdf <= 0);
plt.colorbar()
plt.show()
