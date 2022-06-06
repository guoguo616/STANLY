#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 09:49:40 2022

@author: zjpeters.
"""

# data being read in includes: json, h5, csv, nrrd, jpg, and svg
from skimage import io, filters
from skimage.transform import rescale, rotate
from skimage.exposure import match_histograms
from matplotlib import pyplot as plt
import os
import numpy as np
import json
import csv
import cv2
from glob import glob
import ants
import pandas as pd
from scipy.spatial.distance import pdist, squareform, cosine, cdist
# from sklearn import normalize
# setting up paths
derivatives = "../derivatives"
rawdata = "../rawdata/sleepDepBothBatches"

# need to think about best way to load allen data given the size
# ara_nissl_10 is 10 um, ara_nissl_100 is 100um
# this one takes awhile, so don't rerun often
ara_data = ants.image_read("../data/ccf/ara_nissl_10.nrrd")
annotation_data = ants.image_read("../data/ccf/annotation_10.nrrd")

""" notes about visium data:
    there are a total of 4,992 possible spots on a slide
    tissue_positions_list.csv contains:
    barcode: The sequence of the barcode associated to the spot.
    in_tissue: Binary, indicating if the spot falls inside (1) or outside (0) of tissue.
    array_row: The row coordinate of the spot in the array from 0 to 77. The array has 78 rows.
    array_col: The column coordinate of the spot in the array. In order to express the orange crate arrangement of the spots, this column index uses even numbers from 0 to 126 for even rows, and odd numbers from 1 to 127 for odd rows. Notice then that each row (even or odd) has 64 spots.
    pxl_row_in_fullres: The row pixel coordinate of the center of the spot in the full resolution image.
    pxl_col_in_fullres: The column pixel coordinate of the center of the spot in the full resolution image.


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
"""
end of code from 10x
"""

#%%
# imports the "tissue_hires_image.png" output from spaceranger to register to allen ccf
""" the general workflow should go as follows:
    1. import visium data that has been run through spaceranger pipeline
    2. import relevant atlas images and annotations from allen atlas
    3. prepare visium data for registration into Common Coordinate Framework (ccf)
    4. use SyN registration from ANTs to register visium image to allen image
    5. bring remaining visium data, such as spot coordinates, into allen space using above transformations
    6. measure for nearest neighbor similarity among spots in new space
"""
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
    # the ratio of real spot diameter, 55um, by imaged resolution of spot
    visiumData['sampleStartingResolution'] = 0.55 / visiumData["scaleFactors"]["spot_diameter_fullres"]

    return visiumData

# select which allen slice to align visium data to and import relevant data
def chooseTemplateSlice(sliceLocation):
    templateData = {}
    bestSlice = sliceLocation * 10
    templateSlice = ara_data.slice_image(0,(bestSlice))
    templateAnnotationSlice = annotation_data.slice_image(0,(bestSlice))
    templateLeft = templateSlice[:,570:]
    templateRight = templateSlice[:,:570]
    templateLeft = cv2.normalize(templateLeft, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    templateRight = cv2.normalize(templateRight, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    templateAnnotationLeft = templateAnnotationSlice[:,570:]
    templateAnnotationRight = templateAnnotationSlice[:,:570]
    # templateLeftSliceGauss = filters.gaussian(templateLeftSlice, 10)
    # templateRightSliceGauss = filters.gaussian(templateRightSlice, 10)
    templateData['sliceNumber'] = sliceLocation
    templateData['leftHem'] = templateLeft
    templateData['rightHem'] = templateRight
    templateData['leftHemAnnot'] = templateAnnotationLeft
    templateData['rightHemAnnot'] = templateAnnotationRight
    # currently using the 10um resolution atlas, would need to change if that changes
    templateData['startingResolution'] = 0.01
    
    return templateData

# tissue coordinates should reference output of importVisiumData
# rotation currently accepts 0,90,180,270, will take input from processedVisium
def rotateTissuePoints(visiumData, rotation):
    # scales tissue coordinates down to image resolution
    tissuePointsResizeToHighRes = visiumData["tissuePositionsList"][0:, 3:] * visiumData["scaleFactors"]["tissue_hires_scalef"]
    # below switches x and y in order to properly rotate, this gets undone after registration
    tissuePointsResizeToHighRes[:,[0,1]] = tissuePointsResizeToHighRes[:,[1,0]]  
    # below rotates coordinates and accounts for shift resulting from matrix rotation above, will be different for different angles
    # since the rotation is happening in euclidean space, we have to bring the coordinates back to image space
    if rotation == 0:
        # a null step, but makes for continuous math
        rotMat = [[1,0],[0,1]]
        tissuePointsResizeRotate = np.matmul(tissuePointsResizeToHighRes, rotMat)
        tissuePointsResizeRotate[:,0] = tissuePointsResizeRotate[:,0]
    elif rotation == 90:
        rotMat = [[0,-1],[1,0]]
        tissuePointsResizeRotate = np.matmul(tissuePointsResizeToHighRes, rotMat)
        tissuePointsResizeRotate[:,1] = tissuePointsResizeRotate[:,1] + visiumData["imageData"].shape[1]
    elif rotation == 180:
        rotMat = [[-1,0],[0,-1]]
        tissuePointsResizeRotate = np.matmul(tissuePointsResizeToHighRes, rotMat)
        tissuePointsResizeRotate[:,0] = tissuePointsResizeRotate[:,0] + visiumData["imageData"].shape[1]
        tissuePointsResizeRotate[:,1] = tissuePointsResizeRotate[:,1] + visiumData["imageData"].shape[0]
    elif rotation == 270:
        rotMat = [[0,1],[-1,0]]
        tissuePointsResizeRotate = np.matmul(tissuePointsResizeToHighRes, rotMat)
        tissuePointsResizeRotate[:,0] = tissuePointsResizeRotate[:,0] + visiumData["imageData"].shape[0]
    else:
        print("Incorrect rotation! Please enter: 0, 90, 180, or 270")
    
    return tissuePointsResizeRotate

# prepares visium data for registration
def processVisiumData(visiumData, templateData, rotation):
    processedVisium = {}
    # the sampleID might have issues on non unix given the slash direction, might need to fix
    processedVisium['sampleID'] = visiumData['sampleID']
    processedVisium['tissueSpotBarcodeList'] = visiumData['tissueSpotBarcodeList']
    processedVisium['degreesOfRotation'] = rotation
    processedVisium['visiumGauss'] = filters.gaussian(visiumData['imageData'], sigma=20)
    processedVisium['otsuThreshold'] = filters.threshold_otsu(processedVisium['visiumGauss'])
    processedVisium['visiumOtsu'] = processedVisium['visiumGauss'] < processedVisium['otsuThreshold']
    processedVisium['tissue'] = np.zeros(processedVisium['visiumGauss'].shape)
    processedVisium['tissue'][processedVisium['visiumOtsu']==True] = visiumData['imageData'][processedVisium['visiumOtsu']==True]
    # plt.imshow(visiumData['imageData'])
    # plt.show()
    outputPath = os.path.join(derivatives, visiumData['sampleID'])
    if not os.path.exists(outputPath):
        os.mkdir(outputPath)
    processedVisium['derivativesPath'] = outputPath
    processedVisium['tissueNormalized'] = cv2.normalize(processedVisium['tissue'], None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    processedVisium['resolutionRatio'] = visiumData['sampleStartingResolution'] / templateData['startingResolution']
    processedVisium['tissueResized'] = rescale(processedVisium['tissueNormalized'],processedVisium['resolutionRatio'])
    processedVisium['tissueRotated'] = rotate(processedVisium['tissueResized'], rotation, resize=True)
    processedVisium['tissueHistMatched'] = match_histograms(processedVisium['tissueRotated'], templateData['leftHem'])
    processedVisium['tissuePointsRotated'] = rotateTissuePoints(visiumData, rotation)
    processedVisium['tissuePointsResized'] = processedVisium['tissuePointsRotated'] * processedVisium['resolutionRatio']
    processedVisium['tissuePointsResizedForTransform'] = processedVisium['tissuePointsRotated'] * processedVisium['resolutionRatio']
    processedVisium['tissuePointsResizedForTransform'][:,[0,1]] = processedVisium['tissuePointsResizedForTransform'][:,[1,0]]
    # plt.imshow( processedVisium['tissueRotated'])
    # plt.plot(processedVisium['tissuePointsResized'][:,0],processedVisium['tissuePointsResized'][:,1],marker='.', c='red', alpha=0.2)
    # plt.show()
    processedVisium['filteredFeatureMatrixDense'] = visiumData["filteredFeatureMatrix"][2]
    processedVisium['filteredFeatureMatrixDense'] = processedVisium['filteredFeatureMatrixDense'].todense()
    filteredFeatureMatrixString = []
    for bytebarcode in visiumData['filteredFeatureMatrix'][1]:
        filteredFeatureMatrixString.append(bytebarcode.decode())
    processedVisium['filteredFeatureMatrixBarcodeList'] = filteredFeatureMatrixString 

    filteredFeatureMatrixGeneString = []
    for bytegene in visiumData['filteredFeatureMatrix'][0]['name']:
        filteredFeatureMatrixGeneString.append(bytegene.decode())
    processedVisium['filteredFeatureMatrixGeneList'] = filteredFeatureMatrixGeneString
    header=['x','y','z','t','label','comment']
    csvFormat = []
    rowFormat = []
    with open(f"{os.path.join(outputPath,processedVisium['sampleID'])}_tissuePointsResizeToTemplate.csv", 'w', encoding='UTF8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i in range(len(processedVisium['tissuePointsResizedForTransform'])):
            rowFormat = [processedVisium['tissuePointsResizedForTransform'][i,0]] + [processedVisium['tissuePointsResizedForTransform'][i,1]] + [0] + [0] + [0] + [0]
            writer.writerow(rowFormat)
            csvFormat.append(rowFormat)
            
    processedVisium['tissuePointsForTransform'] = np.array(csvFormat)
    
    # this orders the filtered feature matrix so that the columns are in the order of the coordinate list, so barcodes no longer necessary
    filteredFeatureMatrixBarcodeReorder = []
    for actbarcode in processedVisium['tissueSpotBarcodeList']:
        filteredFeatureMatrixBarcodeReorder.append(processedVisium['filteredFeatureMatrixBarcodeList'].index(actbarcode))
    
    processedVisium['filteredFeatureMatrixOrdered'] = processedVisium['filteredFeatureMatrixDense'][:,filteredFeatureMatrixBarcodeReorder]
    # write re-ordered filtered feature matrix csv to match tissue spot order
    # csvFormat = []
    rowFormat = []
    with open(f"{os.path.join(outputPath,processedVisium['sampleID'])}_tissuePointOrderedFeatureMatrix.csv", 'w', encoding='UTF8') as f:
        writer = csv.writer(f)
        for i in range(len(processedVisium['filteredFeatureMatrixOrdered'])):
            rowFormat = processedVisium['filteredFeatureMatrixOrdered'][i,:]
            writer.writerow(rowFormat)
            # csvFormat.append(rowFormat)
            
    cv2.imwrite(f"{processedVisium['derivativesPath']}/{processedVisium['sampleID']}_tissue.png",processedVisium['tissue'])
    return processedVisium

# think about replacing processedVisium with visiumExperiment that would be like the experiment option below
# will have to add right left hemisphere choice, eventually potentially sagittal etc
# following function registers directly from visium to template
def runANTsToAllenRegistration(processedVisium, templateData):
    # convert into ants image type
    registeredData = {}
    templateAntsImage = ants.from_numpy(templateData['leftHem'])
    sampleAntsImage = ants.from_numpy(processedVisium['tissueHistMatched'])
    synXfm = ants.registration(fixed=templateAntsImage, moving=sampleAntsImage, \
    type_of_transform='SyNAggro', grad_step=0.1, reg_iterations=(120, 100,80,60,40,20,0), \
    syn_sampling=2, flow_sigma=2, syn_metric='mattes', outprefix=os.path.join(processedVisium['derivativesPath'],f"{processedVisium['sampleID']}_xfm"))
    registeredData['antsOutput'] = synXfm
    registeredData['sampleID'] = processedVisium['sampleID']
    registeredData['derivativesPath'] = processedVisium['derivativesPath']
    # apply syn transform to tissue spot coordinates
    os.system(f"antsApplyTransformsToPoints -d 2 -i {os.path.join(processedVisium['derivativesPath'],processedVisium['sampleID'])}_tissuePointsResizeToTemplate.csv -o {os.path.join(processedVisium['derivativesPath'],processedVisium['sampleID'])}_tissuePointsResizeToTemplateTransformApplied.csv -t [ {os.path.join(processedVisium['derivativesPath'],processedVisium['sampleID'])}_xfm0GenericAffine.mat,1] -t {os.path.join(processedVisium['derivativesPath'],processedVisium['sampleID'])}_xfm1InverseWarp.nii.gz")
    
    transformedTissuePositionList = []
    with open(os.path.join(f"{os.path.join(processedVisium['derivativesPath'],processedVisium['sampleID'])}_tissuePointsResizeToTemplateTransformApplied.csv"), newline='') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',')
            next(csvreader)
            for row in csvreader:
                transformedTissuePositionList.append(row)
                
    registeredData['visiumTransformed'] = synXfm["warpedmovout"].numpy()
    registeredData['filteredFeatureMatrixGeneList'] = processedVisium['filteredFeatureMatrixGeneList']
    registeredData['transformedTissuePositionList'] = np.array(transformedTissuePositionList, dtype=float)
    # switching x,y columns back to python compatible and deleting empty columns
    registeredData['transformedTissuePositionList'][:,[0,1]] = registeredData['transformedTissuePositionList'][:,[1,0]]
    registeredData['transformedTissuePositionList'] = np.delete(registeredData['transformedTissuePositionList'], [2,3,4,5],1)

    plt.imshow(registeredData['visiumTransformed'])
    plt.scatter(registeredData['transformedTissuePositionList'][0:,0],registeredData['transformedTissuePositionList'][0:,1], marker='.', c='red', alpha=0.3)
    plt.show()
    
    plt.imshow(registeredData['visiumTransformed'],cmap='gray')
    plt.imshow(templateData['leftHem'], alpha=0.3)
    plt.title(processedVisium['sampleID'])
    plt.show()
        
    transformedTissuePositionListMask = np.logical_and(registeredData['transformedTissuePositionList'] > 0, registeredData['transformedTissuePositionList'] < registeredData['visiumTransformed'].shape[0])
    transformedTissuePositionListFinal = []
    filteredFeatureMatrixBinaryMask = []
    filteredFeatureMatrixMasked = np.zeros(processedVisium['filteredFeatureMatrixOrdered'][:,0].shape)
    for i, masked in enumerate(transformedTissuePositionListMask):
        if masked.all() == True:
            filteredFeatureMatrixBinaryMask.append(1)
            transformedTissuePositionListFinal.append(registeredData['transformedTissuePositionList'][i])
            filteredFeatureMatrixMasked = np.append(filteredFeatureMatrixMasked, processedVisium['filteredFeatureMatrixOrdered'][:,i],axis=1)
        else:
            filteredFeatureMatrixBinaryMask.append(0)
    registeredData['maskedTissuePositionList'] = np.array(transformedTissuePositionListFinal, dtype=float)

    registeredData['filteredFeatureMatrixMasked'] = np.delete(filteredFeatureMatrixMasked, 0,1)
    
    # write re-ordered filtered feature matrix csv to match tissue spot order
    # csvFormat = []
    rowFormat = []
    with open(f"{os.path.join(processedVisium['derivativesPath'],processedVisium['sampleID'])}_tissuePointOrderedFeatureMatrixTemplateMasked.csv", 'w', encoding='UTF8') as f:
        writer = csv.writer(f)
        for i in range(len(registeredData['filteredFeatureMatrixMasked'])):
            rowFormat = registeredData['filteredFeatureMatrixMasked'][i,:]
            writer.writerow(rowFormat)
            # csvFormat.append(rowFormat)
            
    cv2.imwrite(f"{registeredData['derivativesPath']}/{registeredData['sampleID']}_tissue_registered_to_Allen_slice_{templateData['sliceNumber']}.png",registeredData['visiumTransformed'])
    
    return registeredData

def runANTsInterSampleRegistration(processedVisium, sampleToRegisterTo):
    # convert into ants image type
    registeredData = {}
    templateAntsImage = ants.from_numpy(sampleToRegisterTo['tissueHistMatched'])
    sampleAntsImage = ants.from_numpy(processedVisium['tissueHistMatched'])
    synXfm = ants.registration(fixed=templateAntsImage, moving=sampleAntsImage, \
    type_of_transform='SyNAggro', grad_step=0.1, reg_iterations=(100,80,60,40,20,0), \
    syn_sampling=2, flow_sigma=2, syn_metric='mattes', outprefix=os.path.join(processedVisium['derivativesPath'],f"{processedVisium['sampleID']}_to_{sampleToRegisterTo['sampleID']}_xfm"))
    registeredData['antsOutput'] = synXfm
    registeredData['sampleID'] = processedVisium['sampleID']
    registeredData['derivativesPath'] = processedVisium['derivativesPath']
    # ants.plot(templateAntsImage, overlay=synXfm["warpedmovout"])
    # apply syn transform to tissue spot coordinates
    # first line creates a csv file, second line uses that csv as input for antsApplyTransformsToPoints
    # np.savetxt(f"{os.path.join(processedVisium['derivativesPath'],processedVisium['sampleID'])}_tissuePointsResize_to_{sampleToRegisterTo['sampleID']}.csv",processedVisium['tissuePointsForTransform'], delimiter=',', header="x,y,z,t,label,comment")
    os.system(f"antsApplyTransformsToPoints -d 2 -i {os.path.join(processedVisium['derivativesPath'],processedVisium['sampleID'])}_tissuePointsResizeToTemplate.csv -o {os.path.join(processedVisium['derivativesPath'],processedVisium['sampleID'])}_tissuePointsResize_to_{sampleToRegisterTo['sampleID']}TransformApplied.csv -t [ {os.path.join(processedVisium['derivativesPath'],processedVisium['sampleID'])}_to_{sampleToRegisterTo['sampleID']}_xfm0GenericAffine.mat,1] -t {os.path.join(processedVisium['derivativesPath'],processedVisium['sampleID'])}_to_{sampleToRegisterTo['sampleID']}_xfm1InverseWarp.nii.gz")
    
    transformedTissuePositionList = []
    with open(os.path.join(f"{os.path.join(processedVisium['derivativesPath'],processedVisium['sampleID'])}_tissuePointsResize_to_{sampleToRegisterTo['sampleID']}TransformApplied.csv"), newline='') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',')
            next(csvreader)
            for row in csvreader:
                transformedTissuePositionList.append(row)
                
    registeredData['visiumTransformed'] = synXfm["warpedmovout"].numpy()
    registeredData['filteredFeatureMatrixGeneList'] = processedVisium['filteredFeatureMatrixGeneList']

    registeredData['transformedTissuePositionList'] = np.array(transformedTissuePositionList, dtype=float)
    # switching x,y columns back to python compatible and deleting empty columns
    registeredData['transformedTissuePositionList'][:,[0,1]] = registeredData['transformedTissuePositionList'][:,[1,0]]
    registeredData['transformedTissuePositionList'] = np.delete(registeredData['transformedTissuePositionList'], [2,3,4,5],1)
    registeredData['tissueSpotBarcodeList'] = processedVisium["tissueSpotBarcodeList"]
    registeredData['filteredFeatureMatrixOrdered'] = processedVisium['filteredFeatureMatrixOrdered']
    plt.imshow(registeredData['visiumTransformed'])
    plt.scatter(registeredData['transformedTissuePositionList'][0:,0],registeredData['transformedTissuePositionList'][0:,1], marker='.', c='red', alpha=0.3)
    plt.show()
    
    plt.imshow(sampleToRegisterTo['tissueHistMatched'])
    plt.imshow(registeredData['visiumTransformed'], alpha=0.7)
    plt.title(processedVisium['sampleID'])
    plt.show()

    cv2.imwrite(f"{registeredData['derivativesPath']}/{registeredData['sampleID']}_registered_to_{sampleToRegisterTo['sampleID']}.png",registeredData['visiumTransformed'])

    return registeredData
    
def applyAntsTransformations(registeredVisium, bestSampleRegisteredToTemplate, templateData):
    # if not os.exists(f"{os.path.join(registeredVisium['derivativesPath'],registeredVisium['sampleID'])}_tissuePointOrderedFeatureMatrixTemplateMasked.csv"):
        
    templateAntsImage = ants.from_numpy(templateData['leftHem'])
    sampleAntsImage = ants.from_numpy(registeredVisium['visiumTransformed'])
    sampleToTemplate = ants.apply_transforms( fixed=templateAntsImage, moving=sampleAntsImage, transformlist=bestSampleRegisteredToTemplate['antsOutput']['fwdtransforms'])
    
    # make sure this actually does what it's supposed to
    os.system(f"antsApplyTransformsToPoints -d 2 -i {os.path.join(registeredVisium['derivativesPath'],registeredVisium['sampleID'])}_tissuePointsResize_to_{bestSampleRegisteredToTemplate['sampleID']}TransformApplied.csv -o {os.path.join(registeredVisium['derivativesPath'],registeredVisium['sampleID'])}_tissuePointsResize_to_{bestSampleRegisteredToTemplate['sampleID']}TemplateTransformApplied.csv -t [ {os.path.join(bestSampleRegisteredToTemplate['derivativesPath'],bestSampleRegisteredToTemplate['sampleID'])}_xfm0GenericAffine.mat,1] -t {os.path.join(bestSampleRegisteredToTemplate['derivativesPath'],bestSampleRegisteredToTemplate['sampleID'])}_xfm1InverseWarp.nii.gz")
    templateRegisteredData = {}
    transformedTissuePositionList = []
    with open(os.path.join(f"{os.path.join(registeredVisium['derivativesPath'],registeredVisium['sampleID'])}_tissuePointsResize_to_{bestSampleRegisteredToTemplate['sampleID']}TemplateTransformApplied.csv"), newline='') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',')
        next(csvreader)
        for row in csvreader:
            transformedTissuePositionList.append(row)
                
    
    templateRegisteredData['sampleID'] = registeredVisium['sampleID']
    templateRegisteredData['bestFitSampleID'] = bestSampleRegisteredToTemplate['sampleID']
    templateRegisteredData['visiumTransformed'] = sampleToTemplate.numpy()

    templateRegisteredData['transformedTissuePositionList'] = np.array(transformedTissuePositionList, dtype=float)
    # switching x,y columns back to python compatible and deleting empty columns
    templateRegisteredData['transformedTissuePositionList'][:,[0,1]] = templateRegisteredData['transformedTissuePositionList'][:,[1,0]]
    templateRegisteredData['transformedTissuePositionList'] = np.delete(templateRegisteredData['transformedTissuePositionList'], [2,3,4,5],1)
    templateRegisteredData["tissueSpotBarcodeList"] = registeredVisium['tissueSpotBarcodeList']
    templateRegisteredData['filteredFeatureMatrixGeneList'] = registeredVisium['filteredFeatureMatrixGeneList']

    plt.imshow(templateRegisteredData['visiumTransformed'])
    plt.scatter(templateRegisteredData['transformedTissuePositionList'][0:,0],templateRegisteredData['transformedTissuePositionList'][0:,1], marker='.', c='red', alpha=0.3)
    plt.show()
    
    plt.imshow(templateRegisteredData['visiumTransformed'],cmap='gray')
    plt.imshow(templateData['leftHem'], alpha=0.3)
    plt.title(templateRegisteredData['sampleID'])
    plt.show()
        
    transformedTissuePositionListMask = np.logical_and(templateRegisteredData['transformedTissuePositionList'] > 0, templateRegisteredData['transformedTissuePositionList'] < templateRegisteredData['visiumTransformed'].shape[0])
    transformedTissuePositionListFinal = []
    filteredFeatureMatrixBinaryMask = []
    # transformedBarcodesFinal = []
    filteredFeatureMatrixMasked = np.zeros(registeredVisium['filteredFeatureMatrixOrdered'][:,0].shape)
    for i, masked in enumerate(transformedTissuePositionListMask):
        if masked.all() == True:
            filteredFeatureMatrixBinaryMask.append(1)
            transformedTissuePositionListFinal.append(templateRegisteredData['transformedTissuePositionList'][i])
            filteredFeatureMatrixMasked = np.append(filteredFeatureMatrixMasked, registeredVisium['filteredFeatureMatrixOrdered'][:,i],axis=1)
        else:
            filteredFeatureMatrixBinaryMask.append(0)
            # transformedBarcodesFinal.append(templateRegisteredData["tissueSpotBarcodeList"][i])
            
    
    templateRegisteredData['maskedTissuePositionList'] = np.array(transformedTissuePositionListFinal, dtype=float)
    
    templateRegisteredData['filteredFeatureMatrixMasked'] = np.delete(filteredFeatureMatrixMasked,0,1)
    # write re-ordered filtered feature matrix csv to match tissue spot order
    # csvFormat = []
    rowFormat = []
    with open(f"{os.path.join(registeredVisium['derivativesPath'],registeredVisium['sampleID'])}_tissuePointOrderedFeatureMatrixTemplateMasked.csv", 'w', encoding='UTF8') as f:
        writer = csv.writer(f)
        for i in range(len(templateRegisteredData['filteredFeatureMatrixMasked'])):
            rowFormat = templateRegisteredData['filteredFeatureMatrixMasked'][i,:]
            writer.writerow(rowFormat)
            # csvFormat.append(rowFormat)
    cv2.imwrite(f"{os.path.join(registeredVisium['derivativesPath'],registeredVisium['sampleID'])}_registered_to_{bestSampleRegisteredToTemplate['sampleID']}_to_Allen.png",templateRegisteredData['visiumTransformed'])

    return templateRegisteredData

#%% import sample list, location, and degrees of rotation from participants.tsv
# sampleList contains sample ids, templateList contains template slices and degrees of rotation to match
sampleList = []
templateList = []
with open(os.path.join(rawdata,"participants.tsv"), newline='') as tsvfile:
    tsvreader = csv.reader(tsvfile, delimiter='\t')
    next(tsvreader)
    for row in tsvreader:
        sampleList.append(row[0])
        templateList.append(row[1:])

templateList = np.array(templateList, dtype='int')
        
# list of good images
imageList = [0,1, 2, 3, 4, 5, 6, 7, 8, 10, 13]

experiment = {'sample-id': sampleList,
              'template-slice': templateList[:,0],
              'rotation': templateList[:,1],
              'experimental-group': templateList[:,2]}

truncExperiment = {'sample-id': np.asarray(sampleList)[imageList],
                   'template-slice': templateList[imageList,0],
                   'rotation': templateList[imageList,1],
                   'experimental-group': templateList[imageList,2]}

#%% import sample data
# working on below bit
# experimentalResults = dict.fromkeys(['sample-id','antsOutput'])

processedSamples = {}
bestTemplateSlice = 70
template = chooseTemplateSlice(bestTemplateSlice)
for actSample in range(len(truncExperiment['sample-id'])):
    sample = importVisiumData(os.path.join(rawdata, truncExperiment['sample-id'][actSample]))
    # template = chooseTemplateSlice(truncExperiment['template-slice'][actSample])
    
    sampleProcessed = processVisiumData(sample, template, truncExperiment['rotation'][actSample])
    processedSamples[actSample] = sampleProcessed

# in this case just the best looking slice
bestSample = processedSamples[4]

bestSampleToTemplate = runANTsToAllenRegistration(bestSample, template)


#%%
experimentalResults = {}
for actSample in range(len(processedSamples)):
    sampleRegistered = runANTsInterSampleRegistration(processedSamples[actSample], bestSample)
    experimentalResults[actSample] = sampleRegistered


#%%##########################################
# CHECK FOR ACCURACY OF ABOVE REGISTRATIONS #
#############################################
del(ara_data)
del(processedSamples)
#%% 
allSamplesToAllen = {}
for actSample in range(len(experimentalResults)):
    regSampleToTemplate = applyAntsTransformations(experimentalResults[actSample], bestSampleToTemplate, template)
    allSamplesToAllen[actSample] = regSampleToTemplate
    
#%% create digital spots for allen template
# can probably incorporate into import template function
# working on orange crate packing, currently giving roughly 103 spots/mm2 compared to ~118spots/mm2 in original visium slice

# currently working in 10 space, requiring spot coordinates to be divided by 10 at end of calculation
# this is mostly to allow modulo calculation
templateSpots = []
# need to work out the proper scaline, but this is roughly the number of spots/sample as visium slices
################
# tested at 18 #
################
spotDiameter = 18
w = np.sqrt(3) * (spotDiameter/2)   # width of pointy up hexagon
h = spotDiameter    # height of pointy up hexagon
# startingEvenX = spotDiameter/2
# 54.5 is adjusted to make the modulo % work corectly below
# startingOddX = startingEvenX + spotDiameter
# startingEvenY = spotDiameter/2
# startingOddY = startingEvenY + spotDiameter
# templateSpots = [[0], [0]]
currentX = 0
currentY = 0
rowCount = 0

while currentY < template['leftHem'].shape[0]:
    
    if currentX < template['leftHem'].shape[1]:
        templateSpots.append([currentX, currentY])
        currentX += w
    elif (currentX > template['leftHem'].shape[1]):
        # templateSpots.append([currentX, currentY])
        rowCount += 1
        currentY += h * (3/4)
        if ((currentY < template['leftHem'].shape[0]) and (rowCount % 2)):
            currentX = w/2
        else:
            currentX = 0
    elif ((currentX > template['leftHem'].shape[1] * 10) and (currentY > template['leftHem'].shape[0] * 10)):
        print("something is wrong")

templateSpots = np.array(templateSpots)
plt.imshow(template['leftHem'])
plt.scatter(templateSpots[:,0],templateSpots[:,1],alpha=0.3)
plt.show()

# now to remove non-tissue spots
roundedTemplateSpots = np.array(templateSpots.round(), dtype=int)

inTissueTemplateSpots = []
for row in range(len(roundedTemplateSpots)):
    # 15 in the following is just to erode around the edge of the brain
    if template['leftHem'][roundedTemplateSpots[row,1],roundedTemplateSpots[row,0]] > 15:
        inTissueTemplateSpots.append(templateSpots[row])
        
inTissueTemplateSpots = np.array(inTissueTemplateSpots)
plt.imshow(template['leftHem'])
plt.scatter(inTissueTemplateSpots[:,0],inTissueTemplateSpots[:,1], alpha=0.3)
plt.show()

#%% next find nearest neighbor in digital allen spots for each sample spot
# import math # might need math.sqrt
# assuming 1 spot with 6 neighbors

def findDigitalNearestNeighbors(templateSpotsToSearch, templateRegisteredSpots, kNN):
    # finds distance between current spot and list
    allSpotNN = []
    allMeanCdists = []
    for actSpot in templateSpotsToSearch:
        # actSpotNN = []
        # tx, ty = actSpot
        spotCdist = cdist(templateRegisteredSpots, np.array(actSpot).reshape(1,-1), 'euclidean')
        sortedSpotCdist = np.sort(spotCdist, axis=0)
        actSpotCdist = sortedSpotCdist[0:kNN]
        # spotNNIdx gives the index of the top kSpots nearest neighbors for each digital spot
        spotMeanCdist = np.mean(actSpotCdist)
        
        # filledTemplateSpots = []
        spotNNIdx = []
        for i in actSpotCdist:
            if spotMeanCdist < 20:
                actNNIdx = np.where(spotCdist == i)[0]
                spotNNIdx.append(actNNIdx[:])
                # filledTemplateSpots.append(np.array(actSpot))
                # print(actSpot)
            else:
                # should probably change this from 0s to something like -1
                spotNNIdx = np.zeros([kNN,1],dtype=int)
                # spotNNIdx[:] = np.nan
            
        allMeanCdists.append(spotMeanCdist)
        allSpotNN.append(np.array(spotNNIdx))
        
    allSpotNN = np.squeeze(np.array(allSpotNN))
    # should be able to add threshold that removes any spots with a mean cdist > some value
    return allSpotNN

# x = findDigitalNearestNeighbors(inTissueTemplateSpots, allSamplesToAllen[10]['maskedTissuePositionList'])


#%% this line needs to be incorporated into one of the functions, so only run once
for i, regSample in enumerate(allSamplesToAllen):
        
    # removes any spots with fewer than 5000 total gene counts
    countsPerSpot = np.sum(allSamplesToAllen[i]['filteredFeatureMatrixMasked'],axis=0)
    spotMask = countsPerSpot > 5000
    allSamplesToAllen[i]['filteredFeatureMatrixMasked'] = allSamplesToAllen[i]['filteredFeatureMatrixMasked'][:,np.squeeze(np.array(spotMask))]
    allSamplesToAllen[i]['maskedTissuePositionList'] = allSamplesToAllen[i]['maskedTissuePositionList'][np.squeeze(np.array(spotMask)),:]
    # remove genes with no counts
    countsPerGene = np.sum(allSamplesToAllen[i]['filteredFeatureMatrixMasked'],axis=1)
    geneMask = countsPerGene > 0
    allSamplesToAllen[i]['filteredFeatureMatrixMasked'] = allSamplesToAllen[i]['filteredFeatureMatrixMasked'][np.squeeze(np.array(geneMask)),:]
    geneMaskedGeneList = np.array(allSamplesToAllen[i]['filteredFeatureMatrixGeneList'])[np.squeeze(np.array(geneMask))]
    allSamplesToAllen[i]['geneListMasked'] = np.ndarray.tolist(geneMaskedGeneList)
    
#%% compare gene lists and find genes present in all samples
# list of genes present to all slices
allSampleGeneList = allSamplesToAllen[0]['geneListMasked']
for i, regSample in enumerate(allSamplesToAllen):
    if i == 0:
        continue
    allSampleGeneList = set(allSampleGeneList) & set(allSamplesToAllen[i]['geneListMasked'])
        
#%% can now use this gene list to loop over expressed genes 
import scipy
from statsmodels.stats.multitest import multipletests
import matplotlib.colors as mcolors

kSpots = 7
nDigitalSpots = len(inTissueTemplateSpots)
nTotalSamples = len(allSamplesToAllen)
# 'Arc','Egr1','Lars2','Ccl4'
testGeneList = ['Arc','Egr1','Lars2','Ccl4']
caudoputamenGeneList = ['Adora2a','Drd2','Pde10a','Drd1','Scn4b','Gpr6','Ido1','Adcy5','Rasd2','Meis2']
allocortexGeneList = ['Nptxr','Lmo3','Slc30a3','Syn2','Snca','Ccn3','Bmp3','Olfm1','Ldha','Tafa2']
fibertractsGeneList = ['Plp1','Mag','Opalin','Cnp','Trf','Cldn11','Cryab','Mobp','Qdpr','Sept4']
hippocampalregionGeneList = ['Wipf3','Cabp7','Cnih2','Gria1','Ptk2b','Cebpb','Nr3c2','Lct','Arhgef25','Epha7']
hypothalamusGeneList = ['Gpx3','Resp18','AW551984','Minar2','Nap1l5','Gabrq','Pcbd1','Sparc','Vat1','6330403K07Rik']
neocortexGeneList = ['1110008P14Rik','Ccl27a','Mef2c','Tbr1','Cox8a','Snap25','Nrgn','Vxn','Efhd2','Satb2']
striatumlikeGeneList = ['Hap1','Scn5a','Pnck','Ahi1','Snhg11','Galnt16','Pnmal2','Baiap3','Ly6h','Meg3']
thalamusGeneList = ['Plekhg1','Tcf7l2','Ntng1','Ramp3','Rora','Patj','Rgs16','Nsmf','Ptpn4','Rab37']
# testGeneList = caudoputamenGeneList + allocortexGeneList + fibertractsGeneList + hippocampalregionGeneList + hypothalamusGeneList + neocortexGeneList + striatumlikeGeneList + thalamusGeneList
sigGenes = []

for nOfGenesChecked,actGene in enumerate(allSampleGeneList):
    # geneToSearch = actGene
    
    # allSamplesDigitalNearestNeighbors = []
    # digitalSamples = []
    digitalSamplesControl = []
    digitalSamplesExperimental = []
    # meanDigitalSample = np.zeros([nDigitalSpots,1])
    meanDigitalControls = np.zeros([nDigitalSpots])
    meanDigitalExperimentals = np.zeros([nDigitalSpots])
    nTestedSamples = 0
    nControls = 0
    nExperimentals = 0
    for actSample in range(nTotalSamples):
        geneIndex = allSamplesToAllen[actSample]['geneListMasked'].index(actGene)
        spotCheck = np.count_nonzero(allSamplesToAllen[actSample]['filteredFeatureMatrixMasked'][geneIndex,:])
        # use binary mask to remove any tissue spots with no expression
        spotMask = np.transpose(allSamplesToAllen[actSample]['filteredFeatureMatrixMasked'][geneIndex,:] > 0)
        spotMaskedTissuePoints = []
        spotMaskedFeatureMatrix = []
        for idx, actBool in enumerate(spotMask):
            if actBool == True:
                spotMaskedTissuePoints.append(allSamplesToAllen[actSample]['maskedTissuePositionList'][idx,:])
                spotMaskedFeatureMatrix.append(allSamplesToAllen[actSample]['filteredFeatureMatrixMasked'][geneIndex,idx])
        #  an uncomment below to limit search to only those genes with a certain number of expressed spots
        if spotCheck < 15:
            continue
        # print(f"Checking {actGene}")
        spotMaskedTissuePoints = np.array(spotMaskedTissuePoints)
        spotMaskedFeatureMatrix = np.array(spotMaskedFeatureMatrix)
        # actList = allSamplesToAllen[actSample]['maskedTissuePositionList']
        actNN = findDigitalNearestNeighbors(inTissueTemplateSpots, spotMaskedTissuePoints, kSpots)
        # allSamplesDigitalNearestNeighbors.append(actNN)
        geneCount = spotMaskedFeatureMatrix[actNN]
        # digitalSpotGeneCount = []
        for spots in enumerate(actNN):
            if ~np.all(spots[1]):
                geneCount[spots[0]] = np.nan
                
        spotCount = np.nanmean(geneCount, axis=1)
        # digitalSamples.append(spotCount)

        # meanDigitalSample += spotCount
        nTestedSamples += 1
        if truncExperiment['experimental-group'][actSample] == 0:
            # print("Slice is control")
            digitalSamplesControl.append(spotCount)
            meanDigitalControls += spotCount
            # this gives the number of control samples with more than 15 spots containing the gene
            nControls += 1
        elif truncExperiment['experimental-group'][actSample] == 1:
            # print("Slice is experimental")
            digitalSamplesExperimental.append(spotCount)
            meanDigitalExperimentals += spotCount
            # this gives the number of experimental samples with more than 15 spots containing the gene
            nExperimentals += 1
            
        else:
            continue
        
    digitalSamplesControl = np.array(digitalSamplesControl, dtype=float).squeeze()
    digitalSamplesExperimental = np.array(digitalSamplesExperimental, dtype=float).squeeze()
    
    maskedTtests = []
    allTstats = []
    allPvals = []
    for actDigitalSpot in range(nDigitalSpots):
        if ~digitalSamplesControl.any() or ~digitalSamplesExperimental.any():
            continue
        elif (np.sum(np.isnan(digitalSamplesControl[:,actDigitalSpot]) > 3) or (np.sum(np.isnan(digitalSamplesExperimental[:,actDigitalSpot])) > 3)):
            allTstats.append(np.nan)
            allPvals.append(np.nan)
            continue
            
        else:
            # spotMaskCon = np.where(digitalSamplesControl[:,actDigitalSpot] > 0)
            # maskedControlSpots = digitalSamplesControl[digitalSamplesControl[:,actDigitalSpot] > 0, actDigitalSpot]
            # maskedControlSpots = np.array(digitalSamplesControl[spotMaskCon,actDigitalSpot])
            # spotMaskExp = np.where(digitalSamplesExperimental[:,actDigitalSpot] > 0)
            # maskedExperimentalSpots = digitalSamplesExperimental[digitalSamplesExperimental[:,actDigitalSpot] > 0, actDigitalSpot]
            # maskedExperimentalSpots = np.array(digitalSamplesExperimental[spotMaskExp,actDigitalSpot])
            # controlSpots = digitalSamplesControl[:,actDigitalSpot]
            actTtest = scipy.stats.ttest_ind(digitalSamplesControl[:,actDigitalSpot], digitalSamplesExperimental[:,actDigitalSpot],nan_policy='omit')
            allTstats.append(actTtest[0])
            allPvals.append(actTtest[1])
        # if actTtest[1] < 0.05:
        #     maskedTtests.append(actTtest[0])
        # else:
        #     maskedTtests.append(np.nan)
    
    # runs multiple comparisons on pVals for sample
    # mulCompResults = multipletests(allPvals, 0.05, method='fdr_bh')
    # allPvalsSorted = np.sort(allPvals)
    # allPvalsSortedIdx = np.argsort(allPvals)
    pValMask = np.array(allPvals) > 0
    maskedPVals = []
    maskedDigitalCoordinates = []
    maskedMeanDigitalControls = []
    maskedMeanDigitalExperimentals = []
    for idx, actBool in enumerate(pValMask):
        if actBool == True:
            maskedPVals.append(allPvals[idx])
            maskedDigitalCoordinates.append(inTissueTemplateSpots[idx,:])
            maskedMeanDigitalControls.append(meanDigitalControls[idx])
            maskedMeanDigitalExperimentals.append(meanDigitalExperimentals[idx])
            # spotMaskedFeatureMatrix.append(allSamplesToAllen[actSample]['filteredFeatureMatrixMasked'][geneIndex,idx])
            

    if any(maskedPVals):
        maskedDigitalCoordinates = np.array(maskedDigitalCoordinates)
        maskedMeanDigitalControls = np.array(maskedMeanDigitalControls)
        maskedMeanDigitalExperimentals = np.array(maskedMeanDigitalExperimentals)
        mulCompResults = multipletests(maskedPVals, 0.05, method='fdr_bh', is_sorted=False)
        # mulCompResultsAtIdx = np.take_along_axis(np.array(mulCompResults[0]), allPvalsSortedIdx, axis=0)
    else:
        continue
    
    # checkForSigSpots = any(mulCompResultsAtIdx)
    if (any(mulCompResults[0]) == True) & (sum(mulCompResults[0]) > 2):
        finiteMin = np.nanmin(allTstats)
        finiteMax = np.nanmax(allTstats)
        zeroCenteredCmap = mcolors.TwoSlopeNorm(0,vmin=finiteMin, vmax=finiteMax)
        tTestColormap = zeroCenteredCmap(maskedTtests)
        sigGenes.append(actGene)
        # meanDigitalSample = meanDigitalSample / nTestedSamples
        maskedMeanDigitalControls = maskedMeanDigitalControls / nControls
        maskedMeanDigitalExperimentals = maskedMeanDigitalExperimentals / nExperimentals
        maskedFdrTests = []
        for actFdr, test in enumerate(mulCompResults[0]):
            if test == True:
                maskedFdrTests.append(allTstats[actFdr])
            else:
                maskedFdrTests.append(np.nan)
        maxGeneCount = np.nanmax([maskedMeanDigitalControls,maskedMeanDigitalControls])
        # plt.imshow(bestSampleToTemplate['visiumTransformed'])
        # plt.scatter(inTissueTemplateSpots[:,0],inTissueTemplateSpots[:,1], c=np.array(meanDigitalSample), alpha=0.8, vmin=0,vmax=maxGeneCount,plotnonfinite=False)
        # plt.title(f'Mean gene count for {actGene}, all samples')
        # plt.colorbar()
        # plt.show()
        
        plt.imshow(bestSampleToTemplate['visiumTransformed'])
        plt.scatter(maskedDigitalCoordinates[:,0],maskedDigitalCoordinates[:,1], c=np.array(maskedMeanDigitalControls), alpha=0.8, vmin=0,vmax=maxGeneCount,plotnonfinite=False)
        plt.title(f'Mean gene count for {actGene}, controls')
        plt.colorbar()
        plt.show()
        
        plt.imshow(bestSampleToTemplate['visiumTransformed'])
        plt.scatter(maskedDigitalCoordinates[:,0],maskedDigitalCoordinates[:,1], c=np.array(maskedMeanDigitalExperimentals), alpha=0.8, vmin=0,vmax=maxGeneCount,plotnonfinite=False)
        plt.title(f'Mean gene count for {actGene}, experimental')
        plt.colorbar()
        plt.show()
        
        plt.imshow(bestSampleToTemplate['visiumTransformed'])
        plt.scatter(maskedDigitalCoordinates[:,0],maskedDigitalCoordinates[:,1], c=np.array(maskedFdrTests), cmap='seismic',alpha=0.8,norm=zeroCenteredCmap,plotnonfinite=False)
        plt.title(f't-statistic FDR corrected for {actGene}, p < 0.05')
        plt.colorbar()
        plt.show()
    
    
#%% extract gene specific information
# # this version runs ttest on all slices
# import scipy
# from statsmodels.stats.multitest import multipletests
# # Arc index is 25493
# # Vxn, index 27 gives nice cortical spread
# # Sgk3, index 32 seems to be hippocampal
# # Sulf1, index 46, hippocampal `
# # Egr1
# # can search for gene by name, must be an exact match for capitalization
# ################
# # tested at 07 #
# ################
# kSpots = 7
# geneToSearch = 'Ccl4'

# allSamplesDigitalNearestNeighbors = []
# digitalSamples = []
# digitalSamplesControl = []
# digitalSamplesExperimental = []
# meanDigitalSample = np.zeros([len(inTissueTemplateSpots),1])
# meanDigitalControls = np.zeros([len(inTissueTemplateSpots),1])
# meanDigitalExperimentals = np.zeros([len(inTissueTemplateSpots),1])
# nSamples = 0
# nControls = 0
# nExperimentals = 0
# for actSample in range(len(allSamplesToAllen)):
#     geneIndex = allSamplesToAllen[actSample]['geneListMasked'].index(geneToSearch)
#     actList = allSamplesToAllen[actSample]['maskedTissuePositionList']
#     actNN = findDigitalNearestNeighbors(inTissueTemplateSpots, actList, kSpots)
#     allSamplesDigitalNearestNeighbors.append(actNN)
#     geneCount = allSamplesToAllen[actSample]['filteredFeatureMatrixMasked'][geneIndex,actNN]
#     for spots in enumerate(actNN):
#         if ~np.all(spots[1]):
#             geneCount[spots[0]] = np.nan
            
#     spotCount = np.nanmean(geneCount, axis=1)
#     digitalSamples.append(spotCount)
#     meanDigitalSample += spotCount
#     nSamples += 1
#     if truncExperiment['experimental-group'][actSample] == 0:
#         digitalSamplesControl.append(spotCount)
#         meanDigitalControls += spotCount
#         nControls += 1
#     elif truncExperiment['experimental-group'][actSample] == 1:
#         digitalSamplesExperimental.append(spotCount)
#         meanDigitalExperimentals += spotCount
#         nExperimentals += 1
        
# digitalSamplesControl = np.array(digitalSamplesControl, dtype=float).squeeze()
# digitalSamplesExperimental = np.array(digitalSamplesExperimental, dtype=float).squeeze()

# maskedTtests = []
# allTstats = []
# allPvals = []
# for actDigitalSpot in range(len(inTissueTemplateSpots)):
#     spotMaskCon = np.where(digitalSamplesControl[:,actDigitalSpot] > 0)
#     maskedControlSpots = np.array(digitalSamplesControl[spotMaskCon,actDigitalSpot])
#     spotMaskExp = np.where(digitalSamplesExperimental[:,actDigitalSpot] > 0)
#     maskedExperimentalSpots = np.array(digitalSamplesExperimental[spotMaskExp,actDigitalSpot])
#     actTtest = scipy.stats.ttest_ind(np.squeeze(maskedControlSpots), np.squeeze(maskedExperimentalSpots))
#     allTstats.append(actTtest[0])
#     allPvals.append(actTtest[1])
#     if actTtest[1] < 0.05:
#         maskedTtests.append(actTtest[0])
#     else:
#         maskedTtests.append(np.nan)
        
# meanDigitalSample = meanDigitalSample / nSamples
# meanDigitalControls = meanDigitalControls / nControls
# meanDigitalExperimentals = meanDigitalExperimentals / nExperimentals
# mulCompResults = multipletests(allPvals, 0.05, method='fdr_bh')
# maskedTests = []

# checkForSigSpots = any(mulCompResults[0])
# for i, test in enumerate(mulCompResults[0]):
#     if test == True:
#         maskedTests.append(allTstats[i])
#     else:
#         maskedTests.append(np.nan)

#%% extract atlas information
# from allensdk.core.reference_space_cache import ReferenceSpaceCache
# reference_space_key = 'annotation/ccf_2017'
# resolution = 10
# rspc = ReferenceSpaceCache(resolution, reference_space_key, manifest='manifest.json')
# rsp = rspc.get_reference_space()
#%%
# ID 1 is the adult mouse structure graph
# tree = rspc.get_structure_tree(structure_graph_id=1) 
# regionList = tree.get_name_map()
# hippocampus = tree.get_structures_by_name(['Hippocampal formation'])

# hippocampalMask = rsp.make_structure_mask([hippocampus[0]['id']])
# # testingTemplate = template
# # hippocampalMask = np.zeros(template['leftHemAnnot'].shape)
# # for i in hippocampus[0]['structure_id_path']:
# #     hippocampalMask += np.where(testingTemplate['leftHemAnnot'] == i, 1,0)
# #     print(i)
# #     # print("next")
# #     # hippocampalMask[x[0],x[1]] = 1
    

# plt.imshow(hippocampalMask[700,:,570:])
#%%

# bestTemplateSlice10 = bestTemplateSlice * 10
# plt.imshow(bestSampleToTemplate['visiumTransformed'])
# plt.imshow(hippocampalMask[700,:,570:], alpha=0.3)
# plt.show()
#%% plotting statistics
# import matplotlib.colors as mcolors

# finiteMin = np.min(np.array(maskedTtests)[np.isfinite(maskedTtests)])
# finiteMax = np.max(np.array(maskedTtests)[np.isfinite(maskedTtests)])
# zeroCenteredCmap = mcolors.TwoSlopeNorm(0,vmin=finiteMin, vmax=finiteMax)
# tTestColormap = zeroCenteredCmap(maskedTtests)

# plt.imshow(bestSampleToTemplate['visiumTransformed'])
# plt.scatter(inTissueTemplateSpots[:,0],inTissueTemplateSpots[:,1], c=np.array(meanDigitalSample), alpha=0.8)
# plt.title(f'Mean gene count for {geneToSearch}, all samples')
# plt.colorbar()
# plt.show()

# plt.imshow(bestSampleToTemplate['visiumTransformed'])
# plt.scatter(inTissueTemplateSpots[:,0],inTissueTemplateSpots[:,1], c=np.array(meanDigitalControls), alpha=0.8)
# plt.title(f'Mean gene count for {geneToSearch}, controls')
# plt.colorbar()
# plt.show()

# plt.imshow(bestSampleToTemplate['visiumTransformed'])
# plt.scatter(inTissueTemplateSpots[:,0],inTissueTemplateSpots[:,1], c=np.array(meanDigitalExperimentals), alpha=0.8)
# plt.title(f'Mean gene count for {geneToSearch}, experimental')
# plt.colorbar()
# plt.show()

# plt.imshow(bestSampleToTemplate['visiumTransformed'])
# plt.scatter(inTissueTemplateSpots[:,0],inTissueTemplateSpots[:,1], c=np.array(maskedTests), cmap='seismic',alpha=0.8,norm=zeroCenteredCmap,plotnonfinite=False)
# plt.title(f't-statistic for {geneToSearch}, p < 0.05')
# plt.colorbar()
# plt.show()

#%% 
# plt.imshow(bestSample['tissueHistMatched'])
# plt.imshow(experimentalResults[8]['visiumTransformed'],alpha=0.5)
#%% process filtered feature matrix data

# testSample = allSamplesToAllen[0]
# for i, regSample in enumerate(allSamplesToAllen):
        
#     # removes any spots with fewer than 5000 total gene counts
#     countsPerSpot = np.sum(allSamplesToAllen[i]['filteredFeatureMatrixMasked'],axis=0)
#     spotMask = countsPerSpot > 5000
#     allSamplesToAllen[i]['filteredFeatureMatrixMasked'] = allSamplesToAllen[i]['filteredFeatureMatrixMasked'][:,np.squeeze(np.array(spotMask))]
#     allSamplesToAllen[i]['maskedTissuePositionList'] = allSamplesToAllen[i]['maskedTissuePositionList'][np.squeeze(np.array(spotMask)),:]
#     # remove genes with no counts
#     countsPerGene = np.sum(allSamplesToAllen[i]['filteredFeatureMatrixMasked'],axis=1)
#     geneMask = countsPerGene > 0
#     allSamplesToAllen[i]['filteredFeatureMatrixMasked'] = allSamplesToAllen[i]['filteredFeatureMatrixMasked'][np.squeeze(np.array(geneMask)),:]
#     geneMaskedGeneList = np.array(allSamplesToAllen[i]['filteredFeatureMatrixGeneList'])[np.squeeze(np.array(geneMask))]
#     allSamplesToAllen[i]['geneListMasked'] = geneMaskedGeneList
