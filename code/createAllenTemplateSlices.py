#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 11:36:26 2022

@author: zjpeters
"""
# output sample images for user to scroll through in template selection
import ants
from matplotlib import pyplot as plt
import cv2

ara_data = ants.image_read("../data/ccf/ara_nissl_10.nrrd")
# size of data is 1320 in 0 dimension
x = range(10,1321,10)
# this outputs greyscaled versions of the allen slices, converted to 1:255 to use as pngs

for i in x:
    filename = f"allen10umRHSlice{str(i).rjust(4,'0')}.png"
    tempImage = ara_data.slice_image(0,i)
    tempImageLeft = tempImage[:,:570]
    tempImageLeftScaled = (tempImageLeft - tempImageLeft.min())/(tempImageLeft.max() - tempImageLeft.min()) * 255
    # tempImageLeft = cv2.normalize(tempImageLeft, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    # plt.imshow(tempImageLeft)    
    cv2.imwrite(filename, tempImageLeftScaled)
    # ants.image_write(tempImageLeft, filename, ri=True)
    