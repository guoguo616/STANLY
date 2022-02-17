#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 09:49:40 2022

@author: zjpeters
"""
#import skimage
from skimage import io, filters, color
from skimage.transform import rescale, resize, downscale_local_mean, rotate
from matplotlib import pyplot as plt
import os
import numpy as np
import json
import csv
import cv2
#import h5py
import get_matrix_from_h5
import glob
import nrrd
# setting up paths
# needs to read json, h5, jpg, and svg
derivatives = "../derivatives"
rawdata = "../rawdata"

def importVisiumData(sampleFolder):
    # this currently assumes that sampleFolder contains spatial folder and the
    # filtered_feature_bc_matrix.h5 output from space ranger
    visiumData = {}
    visiumData['imageData'] = io.imread(os.path.join(sampleFolder,"spatial","tissue_hires_image.png"), as_gray=True)
    
    tissuePositionsList = []
    with open(os.path.join(sampleFolder,"spatial","tissue_positions_list.csv"), newline='') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',')
        for row in csvreader:
            # checks for in tissue spots
            if row[1] == '1':
                tissuePositionsList.append(row)
    tissuePositionsList = np.asarray(tissuePositionsList)
    visiumData['tissuePositionsList'] = tissuePositionsList
    scaleFactorPath = open(os.path.join(sampleFolder,"spatial","scalefactors_json.json"))
    visiumData['scaleFactors'] = json.loads(scaleFactorPath.read())

    return visiumData

def processVisiumData(visiumSample):
    processedVisium = {}
    processedVisium['visiumGauss'] = filters.gaussian(visiumSample['imageData'], sigma=20)
    processedVisium['otsuThresh'] = filters.threshold_otsu(processedVisium['visiumGauss'])
    processedVisium['visiumOtsu'] = processedVisium['visiumGauss'] < processedVisium['otsuThresh']
    processedVisium['tissue'] = np.zeros(processedVisium['visiumGauss'].shape)
    processedVisium['tissue'][processedVisium['visiumOtsu']==True] = visiumSample['imageData'][processedVisium['visiumOtsu']==True]
    visiumRotate = rotate(processedVisium['tissue'], 180)
    io.imshow(visiumSample['imageData'])
    plt.show()
    io.imshow(processedVisium['visiumGauss'])
    plt.show()
    io.imshow(processedVisium['tissue'])
    return processedVisium
#%%
#filteredFeatureMatrixPath = glob(os.path.join(sampleFolder,"*_filtered_feature_bc_matrix.h5"))
#filteredFeatureMatrix = get_matrix_from_h5.get_matrix_from_h5(filteredFeatureMatrixPath)

# split template image into left and right hemisphere
# the 200 extra pixels in the calculations are just to adjust for centerline, can be removed if needed
templateGrey = io.imread("/media/zjpeters/Samsung_T5/visiumAlignment/data/spatial/mouse_coronal_nissl_slice_73.jpg", as_gray=True)

templateLeft = templateGrey[:,:(templateGrey.shape[1]//2 - 200)]
templateRight = templateGrey[:,(templateGrey.shape[1]//2 + 200):]

templateLeftResize = downscale_local_mean(templateLeft, (4,4))
templateRightResize = downscale_local_mean(templateRight, (4,4))
templateLeftHeavyGauss = filters.gaussian(templateLeftResize,sigma=60)
templateRightHeavyGauss = filters.gaussian(templateRightResize,sigma=60)
templateLeftThresh = filters.threshold_otsu(templateLeftHeavyGauss)
templateRightThresh = filters.threshold_otsu(templateRightHeavyGauss)
templateLeftMask = templateLeftHeavyGauss < templateLeftThresh
templateRightMask = templateRightHeavyGauss < templateRightThresh
#templateLeftLightGauss = filters.gaussian(templateLeftResize,sigma=20)
#templateRightLightGauss = filters.gaussian(templateRightResize,sigma=20)
templateLeftTissue = np.zeros(templateLeftResize.shape)
templateRightTissue = np.zeros(templateRightResize.shape)
templateLeftTissue[templateLeftMask==True] = templateLeftResize[templateLeftMask==True]
templateRightTissue[templateRightMask==True] = templateRightResize[templateRightMask==True] 

#%% work on visium image
# threshold visium image
sample01 = importVisiumData("../rawdata/sleepDepBothBatches/sample-01")

sample01processed = processVisiumData(sample01)

#%%
ara_data, ara_header = nrrd.read("../data/ccf/ara_nissl_100.nrrd")

#%% need to rework this a bit to make more sense comparison-wise
padSizeVert = int((visiumGrey.shape[0]-templateLeftTissue.shape[0])/2)
padSizeHor = int((visiumGrey.shape[1]-templateLeftTissue.shape[1])/2)
templateLeftPad = np.pad(templateLeftTissue, ((padSizeVert,padSizeVert),(padSizeHor,padSizeHor)))
templateRightPad = np.pad(templateRightTissue, ((padSizeVert,padSizeVert),(padSizeHor,padSizeHor)))
io.imshow(templateLeftPad)
#io.imshow(templateRightPad)


#%% work on the dot coordinates
# rotMat contains the matrix to rotate coordinates 180 degrees
rotMat = np.matrix([[0,1],[1,0]])
tissuePositionsListResize = []

for rows in tissuePositionsList:
    tissuePositionsListResize.append([int(rows[4]),int(rows[5])])

tissuePositionsListResize = np.asarray(tissuePositionsListResize)
tissuePositionsListResize[:,0] = np.squeeze(np.floor(tissuePositionsListResize[:,0] * scaleFactors["tissue_hires_scalef"]))
tissuePositionsListResize[:,1] = np.squeeze(np.floor(tissuePositionsListResize[:,1] * scaleFactors["tissue_hires_scalef"]))

tissuePositionsListRotate = np.matmul(tissuePositionsListResize,rotMat)


antsArray = np.zeros([tissuePositionsListRotate.shape[0],4])
antsArray[:,0] = np.squeeze(tissuePositionsListRotate[:,0])
antsArray[:,1] = np.squeeze(tissuePositionsListRotate[:,1])
#antsArray[:,4] = tissuePositionsList[:,0]


#inTissueDots = np.asarray(inTissueDots)
#inTissueDotsResize = np.rint(inTissueDots * scaleFactors['tissue_hires_scalef'])
#inTissueDotsRotate = np.matmul(inTissueDotsResize, rotMat)

#plt.scatter(inTissueDotsRotate[:,1],inTissueDotsRotate[:,0])
#
#io.imsave(os.path.join(derivativesPath,"allenSlice73.png"),np.uint8(templateLeftPad * 255))
#io.imsave(os.path.join(derivativesPath,"sorSlice1.png"),np.uint8(visiumRotate * 255))

#%% work on writing the in tissue coordinates to csv compatible with antsApplyTransformsToPoints

csvFilename = "inTissueDotsRotated.csv"
csvHeader="x,y,z,t"

np.savetxt(csvFilename, antsArray, delimiter=',', header=csvHeader, comments='')



#%%

warpedTissuePositions = []
with open(os.path.join(derivativesPath,"test.csv"), newline='') as csvfile:
    csvreader = csv.reader(csvfile, delimiter=',')
    for row in csvreader:
        warpedTissuePositions.append(row)
        
warpedTissuePositions = np.delete(warpedTissuePositions,(0),axis=0)


#%% plotting spots over visium image
        
fig, ax = plt.subplots()

ax.imshow(visiumGrey)
ax.plot(antsArray[:,0],antsArray[:,1])
#ax.imshow(visiumGrey)ax
#ax.plot(warpedTissuePositions[:,0],warpedTissuePositions[:,1])
#
fig,ax = plt.subplots()
ax.imshow(visiumRotate)
ax.plot(tissuePositionsListResize[:,0],tissuePositionsListResize[:,1])
#
#
#
#
#
#
#
#



