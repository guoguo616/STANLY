#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 09:49:40 2022

@author: zjpeters
"""
#import skimage
from skimage import io, filters, color
from skimage.transform import rescale, resize, downscale_local_mean, rotate
from skimage import exposure
from skimage.exposure import match_histograms
from matplotlib import pyplot as plt
import os
import numpy as np
import json
import csv
import cv2
#import h5py
import get_matrix_from_h5
from glob import glob
import nrrd
import ants
# setting up paths
# needs to read json, h5, jpg, and svg
derivatives = "../derivatives"
rawdata = "../rawdata"

# imports the "tissue_hires_image.png" output to register to allen ccf
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
                tissuePositionsList.append(row[1:])
    tissuePositionsList = np.array(tissuePositionsList, dtype=float)
    visiumData['tissuePositionsList'] = tissuePositionsList
    scaleFactorPath = open(os.path.join(sampleFolder,"spatial","scalefactors_json.json"))
    visiumData['scaleFactors'] = json.loads(scaleFactorPath.read())
#    filteredFeatureMatrixPath = glob(os.path.join(sampleFolder,"*_filtered_feature_bc_matrix.h5"))
#    filteredFeatureMatrix = get_matrix_from_h5.get_matrix_from_h5(filteredFeatureMatrixPath)

    return visiumData

# prepares visium data for registration
def processVisiumData(visiumSample):
    processedVisium = {}
    processedVisium['visiumGauss'] = filters.gaussian(visiumSample['imageData'], sigma=20)
    processedVisium['otsuThresh'] = filters.threshold_otsu(processedVisium['visiumGauss'])
    processedVisium['visiumOtsu'] = processedVisium['visiumGauss'] < processedVisium['otsuThresh']
    processedVisium['tissue'] = np.zeros(processedVisium['visiumGauss'].shape)
    processedVisium['tissue'][processedVisium['visiumOtsu']==True] = visiumSample['imageData'][processedVisium['visiumOtsu']==True]
#    visiumRotate = rotate(processedVisium['tissue'], 180)
    io.imshow(visiumSample['imageData'])
    plt.show()
    io.imshow(processedVisium['visiumGauss'])
    plt.show()
    io.imshow(processedVisium['tissue'])
    return processedVisium
#(act - min(act(:))) / (max(act(:)) - min(act(:)));
# select which allen slice to align visium data to and import relevant data
def chooseTemplateSlice(sliceLocation, resolution):
    if resolution == 100:
        ara_data = ants.image_read("../data/ccf/ara_nissl_100.nrrd")
#        ara_min = np.min(ara_data[np.nonzero(ara_data)])
#        ara_max = ara_data.max()
        annotation_data = ants.image_read("../data/ccf/annotation_100.nrrd")
        templateSlice = ara_data[sliceLocation,:,:]
        templateAnnotationSlice = annotation_data[sliceLocation,:,:]
#        
#        templateHeader["dimension"] = 2
#        templateHeader["sizes"] = [templateSlice.shape[0], templateSlice.shape[1]]
#        templateHeader["space directions"] = templateHeader["space directions"][0:2,0:2]
#        templateHeader["space origin"] = [0., 0.]
        templateLeftSlice = templateSlice[:,0:templateSlice.shape[1]//2]
        templateRightSlice = templateSlice[:,(templateSlice.shape[1]//2+1):]
        templateStartingResolution = 0.1
#        templateLeftSlice = templateLeftSlice/ara_max
        io.imshow(templateLeftSlice)
    elif resolution == 10:
        ara_data = ants.image_read("../data/ccf/ara_nissl_10.nrrd")

        annotation_data = ants.image_read("../data/ccf/annotation_10.nrrd")
        sliceLocation10um = sliceLocation * 10
        templateSlice = ara_data[sliceLocation10um,:,:]
        templateAnnotationSlice = annotation_data[sliceLocation10um,:,:]
#        templateHeader = ara_header
#        templateHeader["dimension"] = 2
#        templateHeader["sizes"] = [templateSlice.shape[0], templateSlice.shape[1]]
#        templateHeader["space directions"] = templateHeader["space directions"][0:2,0:2]
#        templateHeader["space origin"] = [0., 0.] 
        templateLeftSlice = templateSlice[:,0:templateSlice.shape[1]//2]
        templateRightSlice = templateSlice[:,(templateSlice.shape[1]//2+1):]
        templateLeftSliceGauss = filters.gaussian(templateLeftImage, 10)
        templateRightSliceGauss = filters.gaussian(templateRightImage, 10)
        templateStartingResolution = 0.01
#        templateMin = np.min(templateLeftSlice[np.nonzero(templateLeftSliceGauss)])
#        templateMax = templateLeftSliceGauss.max()
#        templateLeftSlice = np.zeros([templateLeftImage.shape[0],templateLeftImage.shape[1]])
#        templateLeftSlice = ((templateLeftSliceGauss - templateMin) / (templateMax - templateMin))
#        templateLeftSlice[templateLeftSlice < 0] = 0
#        templateRightSlice = np.zeros([templateRightImage.shape[0],templateRightImage.shape[1]])
#        templateRightSlice = ((templateRightSliceGauss - templateMin) / (templateMax - templateMin))
#        templateRightSlice[templateRightSlice < 0] = 0
        io.imshow(templateLeftSlice)
    else:
        print("No matching data of that resolution, please choose: 10 or 100 um:")
    
    
    return templateSlice, templateLeftSlice, templateRightSlice, templateAnnotationSlice, templateStartingResolution
#
#allenSlice73, allenLeftSlice73, allenRightSlice73, allenAnnotSlice73 = chooseTemplateSlice(73, 10)
#%% import allen data
# this one takes awhile, so don't rerun often
ara_data = ants.image_read("../data/ccf/ara_nissl_10.nrrd")
annotation_data = ants.image_read("../data/ccf/annotation_10.nrrd")
#%%, finish "chooseTemplateSlice" from this code
# ara_nissl_10 is 10 um, ara_nissl_100 is 100um
templateStartingResolution = 0.01
templateSlice = ara_data.slice_image(0,650)
templateAnnotationSlice = annotation_data.slice_image(0,660)
#        templateHeader["dimension"] = 2
#        templateHeader["sizes"] = [templateSlice.shape[0], templateSlice.shape[1]]
#        templateHeader["space directions"] = templateHeader["space directions"][0:2,0:2]
#        templateHeader["space origin"] = [0., 0.]
#
#templateLeftSlice = templateSlice[:,0:templateSlice.shape[1]//2]
#templateRightSlice = templateSlice[:,(templateSlice.shape[1]//2+1):]
#        templateLeftSlice = templateLeftSlice/ara_max
#io.imshow(templateLeftSlice)
templateLeft = templateSlice[:,571:]

templateLeft = cv2.normalize(templateLeft, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

#%%

templateLeftGauss = filters.gaussian(templateLeft, 10)
io.imshow(templateLeftGauss)
#%% import sample data
# this resolution is the mm/pixel estimation of the 6.5mm between dot borders, which is ~1560 pixels in the tissue_hires_image.png
sampleStartingResolution = 6.5 / 1560
resolutionRatio = sampleStartingResolution / templateStartingResolution

sample = importVisiumData("../rawdata/sleepDepBothBatches/sample-01")
sampleProcessed = processVisiumData(sample)
#%% 
sampleNorm = cv2.normalize(sampleProcessed["tissue"], None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
# (np.sum((templateLeft > 0)/np.sum(sampleNorm > 0))) for below
degreesToRotate = -90
sampleResize = rescale(sampleNorm,resolutionRatio)
# if resize=True not set, image will be slightly misaligned with spots
sampleRotate = rotate(sampleResize, degreesToRotate, resize=True)
sampleHistMatch = match_histograms(sampleRotate, templateLeft)

#%%

templateAntsImage = ants.from_numpy(templateLeft)
sampleAntsImage = ants.from_numpy(sampleHistMatch)
templateAntsImage.set_spacing([templateStartingResolution,templateStartingResolution])
sampleAntsImage.set_spacing([templateStartingResolution,templateStartingResolution])

synXfm = ants.registration(fixed=templateAntsImage, moving=sampleAntsImage, type_of_transform='SyN', outprefix='../derivatives/sample-01_xfm')
ants.plot(templateAntsImage, overlay=affineXfm["warpedmovout"])

#%% affine alignment of points

#%% Tissue point registration
#rotMat = [[np.cos(degreesToRotate),-np.sin(degreesToRotate)],[np.sin(degreesToRotate),np.cos(degreesToRotate)]]
# -90 = [[0,1],[-1,0]]
rotMat = [[0,1],[-1,0]]

tissuePointsResizeToHighRes = sample["tissuePositionsList"][0:, 3:] * sample["scaleFactors"]["tissue_hires_scalef"]
tissuePointsResizeToHighRes[:,[0,1]] = tissuePointsResizeToHighRes[:,[1,0]]
plt.imshow(sampleProcessed["tissue"])
plt.plot(tissuePointsResizeToHighRes[:,0],tissuePointsResizeToHighRes[:,1],marker="o",color="blue")
plt.show()

tissuePointsResizeRotate = np.matmul(tissuePointsResizeToHighRes, rotMat)
# below accounts for shift resulting from matrix rotation above, will be different for different angles
tissuePointsResizeRotate[:,0] = tissuePointsResizeRotate[:,0] + sampleProcessed["tissue"].shape[0]
tissuePointsResizeToTemplate = tissuePointsResizeRotate * resolutionRatio
plt.imshow(sampleRotate)
plt.plot(tissuePointsResizeToTemplate[:,0],tissuePointsResizeToTemplate[:,1],marker='o',color='red')
plt.show()

#%% apply syn transform to tissue spot coordinates
csvPad = np.zeros(tissuePointsResizeToTemplate.shape)
tissuePointsForTransform = np.append(tissuePointsResizeToTemplate, csvPad, 1)
np.savetxt('../derivatives/sample-01_tissuePointsResizeToTemplate.csv',tissuePointsForTransform, delimiter=',', header="x,y,z,t")
os.system("antsApplyTransformsToPoints -d 2 -i ../derivatives/sample-01_tissuePointsResizeToTemplate.csv -o ../derivatives/sample-01_tissuePointsResizeToTemplateTransformApplied.csv -t [ ../derivatives/sample-01_xfm0GenericAffine.mat,1]")

#%% open and check transformed coordinates
transformedTissuePositionList = []
with open(os.path.join('../derivatives/sample-01_tissuePointsResizeToTemplateTransformApplied.csv'), newline='') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',')
        next(csvreader)
        for row in csvreader:
            transformedTissuePositionList.append(row)
            
sampleTransformed = synXfm["warpedmovout"].numpy()

transformedTissuePositionList = np.array(transformedTissuePositionList, dtype=float)

plt.imshow(sampleTransformed)
plt.plot(transformedTissuePositionList[0:,0],transformedTissuePositionList[0:,1], marker='o', color='green')
plt.show()






