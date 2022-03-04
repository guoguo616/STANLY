#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 09:49:40 2022

@author: zjpeters
"""
#import skimage
from skimage import io, filters
from skimage.transform import rescale, rotate
# from skimage import exposure
from skimage.exposure import match_histograms
from matplotlib import pyplot as plt
import os
import numpy as np
import json
import csv
import cv2
#import h5py
#mport get_matrix_from_h5
from glob import glob
import ants
from allensdk.core.reference_space_cache import ReferenceSpaceCache
"""
get_matrix_from_h5 is from code @:
    https://support.10xgenomics.com/spatial-gene-expression/software/pipelines/latest/advanced/h5_matrices
"""
import collections
import scipy.sparse as sp_sparse
import tables

CountMatrix = collections.namedtuple('CountMatrix', ['feature_ref', 'barcodes', 'matrix'])
 
def get_matrix_from_h5(filename):
    with tables.open_file(filename, 'r') as f:
        mat_group = f.get_node(f.root, 'matrix')
        barcodes = f.get_node(mat_group, 'barcodes').read()
        data = getattr(mat_group, 'data').read()
        indices = getattr(mat_group, 'indices').read()
        indptr = getattr(mat_group, 'indptr').read()
        shape = getattr(mat_group, 'shape').read()
        matrix = sp_sparse.csc_matrix((data, indices, indptr), shape=shape)
         
        feature_ref = {}
        feature_group = f.get_node(mat_group, 'features')
        feature_ids = getattr(feature_group, 'id').read()
        feature_names = getattr(feature_group, 'name').read()
        feature_types = getattr(feature_group, 'feature_type').read()
        feature_ref['id'] = feature_ids
        feature_ref['name'] = feature_names
        feature_ref['feature_type'] = feature_types
        tag_keys = getattr(feature_group, '_all_tag_keys').read()
        for key in tag_keys:
            feature_ref[key] = getattr(feature_group, key.decode('UTF-8')).read()
#         
        return CountMatrix(feature_ref, barcodes, matrix)

# setting up paths
# needs to read json, h5, jpg, and svg
derivatives = "../derivatives"
rawdata = "../rawdata"


# import allen data finish "chooseTemplateSlice" from this code
# ara_nissl_10 is 10 um, ara_nissl_100 is 100um
# this one takes awhile, so don't rerun often
ara_data = ants.image_read("../data/ccf/ara_nissl_10.nrrd")
annotation_data = ants.image_read("../data/ccf/annotation_10.nrrd")
#%%
# imports the "tissue_hires_image.png" output to register to allen ccf
def importVisiumData(sampleFolder):
    # this currently assumes that sampleFolder contains spatial folder and the
    # filtered_feature_bc_matrix.h5 output from space ranger
    visiumData = {}
    visiumData['imageData'] = io.imread(os.path.join(sampleFolder,"spatial","tissue_hires_image.png"), as_gray=True)
    visiumData['sampleID'] = sampleFolder.rsplit(sep='/',maxsplit=1)[-1]
    tissuePositionsList = []
    tissueSpotBarcodes = []
    with open(os.path.join(sampleFolder,"spatial","tissue_positions_list.csv"), newline='') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',')
        for row in csvreader:
            # checks for in tissue spots
            if row[1] == '1':
                tissueSpotBarcodes.append(row[0])
                tissuePositionsList.append(row[1:])
    tissuePositionsList = np.array(tissuePositionsList, dtype=float)
    visiumData['tissueSpotBarcodeList'] = tissueSpotBarcodes
    visiumData['tissuePositionsList'] = tissuePositionsList
    scaleFactorPath = open(os.path.join(sampleFolder,"spatial","scalefactors_json.json"))
    visiumData['scaleFactors'] = json.loads(scaleFactorPath.read())
    scaleFactorPath.close()
    filteredFeatureMatrixPath = glob(os.path.join(sampleFolder,"*_filtered_feature_bc_matrix.h5"))
    filteredFeatureMatrix = get_matrix_from_h5(os.path.join(filteredFeatureMatrixPath[0]))
    visiumData['filteredFeatureMatrix'] = filteredFeatureMatrix

    return visiumData

# prepares visium data for registration
def processVisiumData(visiumSample):
    processedVisium = {}
    # the sampleID might have issues on non unix given the slash direction, might need to fix
    processedVisium['visiumGauss'] = filters.gaussian(visiumSample['imageData'], sigma=20)
    processedVisium['otsuThresh'] = filters.threshold_otsu(processedVisium['visiumGauss'])
    processedVisium['visiumOtsu'] = processedVisium['visiumGauss'] < processedVisium['otsuThresh']
    processedVisium['tissue'] = np.zeros(processedVisium['visiumGauss'].shape)
    processedVisium['tissue'][processedVisium['visiumOtsu']==True] = visiumSample['imageData'][processedVisium['visiumOtsu']==True]
    io.imshow(visiumSample['imageData'])
    plt.show()
    io.imshow(processedVisium['tissue'])
    outputPath = os.path.join(derivatives, visiumSample['sampleID'])
    if not os.path.exists(outputPath):
        os.mkdir(outputPath)
    processedVisium['derivativesPath'] = outputPath
    return processedVisium

# select which allen slice to align visium data to and import relevant data
def chooseTemplateSlice(sliceLocation, resolution):
    if resolution == 100:
        ara_data = ants.image_read("../data/ccf/ara_nissl_100.nrrd")

        annotation_data = ants.image_read("../data/ccf/annotation_100.nrrd")
        templateSlice = ara_data[sliceLocation,:,:]
        templateAnnotationSlice = annotation_data[sliceLocation,:,:]
        templateLeftSlice = templateSlice[:,0:templateSlice.shape[1]//2]
        templateRightSlice = templateSlice[:,(templateSlice.shape[1]//2+1):]
        templateStartingResolution = 0.1
        # io.imshow(templateLeftSlice)
    elif resolution == 10:
        ara_data = ants.image_read("../data/ccf/ara_nissl_10.nrrd")

        annotation_data = ants.image_read("../data/ccf/annotation_10.nrrd")
        sliceLocation10um = sliceLocation * 10
        templateSlice = ara_data[sliceLocation10um,:,:]
        templateAnnotationSlice = annotation_data[sliceLocation10um,:,:]
        templateLeftSlice = templateSlice[:,0:templateSlice.shape[1]//2]
        templateRightSlice = templateSlice[:,(templateSlice.shape[1]//2+1):]
        templateLeftSliceGauss = filters.gaussian(templateLeftSlice, 10)
        templateRightSliceGauss = filters.gaussian(templateRightSlice, 10)
        templateStartingResolution = 0.01
        # io.imshow(templateLeftSlice)
    else:
        print("No matching data of that resolution, please choose: 10 or 100 um:")
    
    
    return templateSlice, templateLeftSlice, templateRightSlice, templateAnnotationSlice, templateStartingResolution
#
#allenSlice73, allenLeftSlice73, allenRightSlice73, allenAnnotSlice73 = chooseTemplateSlice(73, 10)

#%% choose template slice

# list of best template slices for each sample could be a good inclusion in participants.tsv, if using format
# sample-01: 68, sample-02: 66, sample-03: na, sample-04: 72, sample-05 72, sample-06: 66
bestSlice = 66
# sample-01: 270, sample-02: 270, sample-03: na, sample-04: 0, sample-05: 90, sample-06: 90
# should be able to do away with this by implementing the fiducial alignment
degreesToRotate = 90
# below is set up for 10um
templateStartingResolution = 0.01
templateSlice = ara_data.slice_image(0,(bestSlice*10))
templateAnnotationSlice = annotation_data.slice_image(0,(bestSlice*10))

# this cuts the image in half to use only the left/right hemisphere
templateLeft = templateSlice[:,571:]
templateAnnotationLeft = templateAnnotationSlice[:,571:]
templateLeft = cv2.normalize(templateLeft, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

templateLeftGauss = filters.gaussian(templateLeft, 10)
io.imshow(templateLeftGauss)
#%% import sample data
# this resolution is the mm/pixel estimation of the 6.5mm between dot borders, which is ~1560 pixels in the tissue_hires_image.png
sampleStartingResolution = 6.5 / 1560
resolutionRatio = sampleStartingResolution / templateStartingResolution

sample = importVisiumData("../rawdata/sleepDepBothBatches/sample-07")
sampleProcessed = processVisiumData(sample)
sampleMatrix = sample["filteredFeatureMatrix"][2]
sampleMatrix = sampleMatrix.todense()
#%% 
sampleNorm = cv2.normalize(sampleProcessed["tissue"], None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
# (np.sum((templateLeft > 0)/np.sum(sampleNorm > 0))) for below

sampleResize = rescale(sampleNorm,resolutionRatio)
# if resize=True not set, image will be slightly misaligned with spots
sampleRotate = rotate(sampleResize, degreesToRotate, resize=True)
sampleHistMatch = match_histograms(sampleRotate, templateLeft)

#%% run registration of sample to template

templateAntsImage = ants.from_numpy(templateLeft)
sampleAntsImage = ants.from_numpy(sampleHistMatch)
# templateAntsImage.set_spacing([templateStartingResolution,templateStartingResolution])
# sampleAntsImage.set_spacing([templateStartingResolution,templateStartingResolution])

synXfm = ants.registration(fixed=templateAntsImage, moving=sampleAntsImage, type_of_transform='SyN', outprefix=os.path.join(sampleProcessed['derivativesPath'],f"{sample['sampleID']}_xfm"))
ants.plot(templateAntsImage, overlay=synXfm["warpedmovout"])

#%%
# CHECK FOR ACCURACY OF ABOVE REGISTRATION
#

#%% Tissue point registration
# update tissue points with pre-registration alignment of sample image
# -90 = [[0,1],[-1,0]]
rotMat = [[0,1],[-1,0]]
if degreesToRotate == 0:
    rotMat = [[1,0],[0,1]]
elif degreesToRotate == 90:
    rotMat = [[0,-1],[1,0]]
elif degreesToRotate == 270:
    rotMat = [[0,1],[-1,0]]
tissuePointsResizeToHighRes = sample["tissuePositionsList"][0:, 3:] * sample["scaleFactors"]["tissue_hires_scalef"]
# below switches x and y in order to properly rotate, this gets undone in next cell
tissuePointsResizeToHighRes[:,[0,1]] = tissuePointsResizeToHighRes[:,[1,0]]
plt.imshow(sampleProcessed["tissue"])
plt.plot(tissuePointsResizeToHighRes[:,0],tissuePointsResizeToHighRes[:,1],marker='.', c='blue', alpha=0.2)
plt.show()

tissuePointsResizeRotate = np.matmul(tissuePointsResizeToHighRes, rotMat)
# below accounts for shift resulting from matrix rotation above, will be different for different angles
if degreesToRotate == 0:
    tissuePointsResizeRotate[:,0] = tissuePointsResizeRotate[:,0]
elif degreesToRotate == 90:
    tissuePointsResizeRotate[:,1] = tissuePointsResizeRotate[:,1] + sampleProcessed["tissue"].shape[1]
elif degreesToRotate == 270:
    tissuePointsResizeRotate[:,0] = tissuePointsResizeRotate[:,0] + sampleProcessed["tissue"].shape[0]
tissuePointsResizeToTemplate = tissuePointsResizeRotate * resolutionRatio
plt.imshow(sampleRotate)
plt.plot(tissuePointsResizeToTemplate[:,0],tissuePointsResizeToTemplate[:,1],marker='.', c='red', alpha=0.2)
plt.show()

#%% apply syn transform to tissue spot coordinates

# next line reverses x,y switching of above
tissuePointsResizeToTemplate[:,[0,1]] = tissuePointsResizeToTemplate[:,[1,0]]

csvPad = np.zeros([tissuePointsResizeToTemplate.shape[0],4])
tissuePointsForTransform = np.append(tissuePointsResizeToTemplate, csvPad, 1)
np.savetxt(f"{os.path.join(sampleProcessed['derivativesPath'],sample['sampleID'])}_tissuePointsResizeToTemplate.csv",tissuePointsForTransform, delimiter=',', header="x,y,z,t,label,comment")
os.system(f"antsApplyTransformsToPoints -d 2 -i {os.path.join(sampleProcessed['derivativesPath'],sample['sampleID'])}_tissuePointsResizeToTemplate.csv -o {os.path.join(sampleProcessed['derivativesPath'],sample['sampleID'])}_tissuePointsResizeToTemplateTransformApplied.csv -t [ {os.path.join(sampleProcessed['derivativesPath'],sample['sampleID'])}_xfm0GenericAffine.mat,1] -t {os.path.join(sampleProcessed['derivativesPath'],sample['sampleID'])}_xfm1InverseWarp.nii.gz")

#%% open and check transformed coordinates
transformedTissuePositionList = []
with open(os.path.join(f"{os.path.join(sampleProcessed['derivativesPath'],sample['sampleID'])}_tissuePointsResizeToTemplateTransformApplied.csv"), newline='') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',')
        next(csvreader)
        for row in csvreader:
            transformedTissuePositionList.append(row)
            
sampleTransformed = synXfm["warpedmovout"].numpy()

transformedTissuePositionList = np.array(transformedTissuePositionList, dtype=float)
# again, switching x,y columns back to python compatible
transformedTissuePositionList[:,[0,1]] = transformedTissuePositionList[:,[1,0]]
transformedTissuePositionList = np.delete(transformedTissuePositionList, [2,3,4,5],1)

plt.imshow(sampleTransformed)
plt.scatter(transformedTissuePositionList[0:,0],transformedTissuePositionList[0:,1], marker='.', c='red', alpha=0.3)
plt.show()

#%% remove any out of bounds points and prepare for comparison to atlas locations

# might want to include section that removes any negative coordinates following spot registration
# one other option might be to use the whole brain slice for the annotated extraction

# transformedTissuePositionListMask = transformedTissuePositionList > 0 
transformedTissuePositionListMask = np.logical_and(transformedTissuePositionList > 0, transformedTissuePositionList < sampleTransformed.shape[0])

transformedTissuePositionListFinal = [];
transformedBarcodesFinal = []
for i, masked in enumerate(transformedTissuePositionListMask):
    if masked.all() == True:
        transformedTissuePositionListFinal.append(transformedTissuePositionList[i])
        transformedBarcodesFinal.append(sample["tissueSpotBarcodeList"][i])

transformedTissuePositionListFinal = np.array(transformedTissuePositionListFinal, dtype=float)

plt.imshow(templateLeft)
plt.scatter(transformedTissuePositionListFinal[0:,0],transformedTissuePositionListFinal[0:,1], marker='x', c='red', alpha=0.3)
plt.show()
plt.imshow

# create a "fake" annotation image by replacing all regions with # > 1500 with one value that just looks better in an overlay
templateAnnotationLeftFake = templateAnnotationLeft
templateAnnotationLeftFake[templateAnnotationLeft > 1500] = 100
plt.imshow(templateAnnotationLeft)
plt.scatter(transformedTissuePositionListFinal[0:,0],transformedTissuePositionListFinal[0:,1], marker='.', c='red', alpha=0.5)
plt.show()
plt.imshow

#%%
plt.imshow(sampleTransformed)
plt.scatter(transformedTissuePositionListFinal[0:,0],transformedTissuePositionListFinal[0:,1], marker='x', c='red', alpha=0.3)
plt.show()
plt.imshow

plt.imshow(sampleTransformed,cmap='gray')
plt.imshow(templateAnnotationLeft, alpha=0.3)
plt.show()
#%% extract atlas information
reference_space_key = 'annotation/ccf_2017'
resolution = 10
rspc = ReferenceSpaceCache(resolution, reference_space_key, manifest='manifest.json')
# ID 1 is the adult mouse structure graph
tree = rspc.get_structure_tree(structure_graph_id=1) 
regionList = tree.get_name_map()
hippocampus = tree.get_structures_by_name(['Hippocampal region'])
hippocampus[0]['id']

hippocampalMask = np.zeros(templateAnnotationLeft.shape)
hippocampalMask[templateAnnotationLeft == 1089] = 1
# hippocampalMask[hippocampalMask != 1080 ] = 0
# find regions present in current annotation slice
# templateRegions = np.unique(templateAnnotationLeft)

# for i in templateRegions:
#     if i > 0:
        









