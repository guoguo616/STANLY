#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""".
Created on Sep 7 2022

@author: zjpeters.
"""
# %%
# data being read in includes: json, h5, csv, nrrd, jpg, and svg
from skimage import io, filters, color
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
import scipy.spatial as sp_spatial
import scipy.sparse as sp_sparse
import collections
import tables
# from scipy.spatial.distance import pdist, squareform, cosine, cdist
# setting up paths
derivatives = "../derivatives"
rawdata = "../rawdata/sleepDepBothBatches"

# need to think about best way to load allen data given the size
# ara_nissl_10 is 10 um, ara_nissl_100 is 100um

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
    6. measure for nearest neighbor similarity among spots in new space and create a vector that represents the nearest neighbors from each slice
"""
def importVisiumData(sampleFolder):
    # this currently assumes that sampleFolder contains spatial folder and the
    # filtered_feature_bc_matrix.h5 output from space ranger
    visiumData = {}
    visiumData['imageData'] = io.imread(os.path.join(sampleFolder,"spatial","tissue_hires_image.png"))
    visiumData['imageDataGray'] = color.rgb2gray(visiumData['imageData'])
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
    visiumData['spotStartingResolution'] = 0.55 / visiumData["scaleFactors"]["spot_diameter_fullres"]
    # plt.imshow(visiumData['imageData'])
    return visiumData

# use to select which allen slice to align visium data to and import relevant data
def chooseTemplateSlice(sliceLocation):
    ara_data = ants.image_read("../data/ccf/ara_nissl_10.nrrd")
    annotation_data = ants.image_read("../data/ccf/annotation_10.nrrd")
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
    templateData['templateLeftGauss'] = filters.gaussian(templateLeft, sigma=10)
    templateData['templateRightGauss'] = filters.gaussian(templateRight, sigma=10)
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
        tissuePointsResizeRotate[:,1] = tissuePointsResizeRotate[:,1] + visiumData["imageDataGray"].shape[1]
    elif rotation == 180:
        rotMat = [[-1,0],[0,-1]]
        tissuePointsResizeRotate = np.matmul(tissuePointsResizeToHighRes, rotMat)
        tissuePointsResizeRotate[:,0] = tissuePointsResizeRotate[:,0] + visiumData["imageDataGray"].shape[1]
        tissuePointsResizeRotate[:,1] = tissuePointsResizeRotate[:,1] + visiumData["imageDataGray"].shape[0]
    elif rotation == 270:
        rotMat = [[0,1],[-1,0]]
        tissuePointsResizeRotate = np.matmul(tissuePointsResizeToHighRes, rotMat)
        tissuePointsResizeRotate[:,0] = tissuePointsResizeRotate[:,0] + visiumData["imageDataGray"].shape[0]
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
    processedVisium['visiumGauss'] = filters.gaussian(visiumData['imageDataGray'], sigma=20)
    processedVisium['otsuThreshold'] = filters.threshold_otsu(processedVisium['visiumGauss'])
    processedVisium['visiumOtsu'] = processedVisium['visiumGauss'] < processedVisium['otsuThreshold']
    processedVisium['tissue'] = np.zeros(processedVisium['visiumGauss'].shape)
    processedVisium['tissue'][processedVisium['visiumOtsu']==True] = visiumData['imageDataGray'][processedVisium['visiumOtsu']==True]
    # why do i have the gaussian level switch here? should be consistent
    processedVisium['visiumGauss'] = filters.gaussian(visiumData['imageDataGray'], sigma=5)
    processedVisium['tissueGauss'] = np.zeros(processedVisium['visiumGauss'].shape)
    processedVisium['tissueGauss'][processedVisium['visiumOtsu']==True] = processedVisium['visiumGauss'][processedVisium['visiumOtsu']==True]

    # plt.imshow(visiumData['imageData'])
    # plt.show()
    outputPath = os.path.join(derivatives, visiumData['sampleID'])
    if not os.path.exists(outputPath):
        os.mkdir(outputPath)
    processedVisium['derivativesPath'] = outputPath
    processedVisium['tissueNormalized'] = cv2.normalize(processedVisium['tissueGauss'], None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    processedVisium['resolutionRatio'] = visiumData['spotStartingResolution'] / templateData['startingResolution']
    processedVisium['tissueResized'] = rescale(processedVisium['tissueNormalized'],processedVisium['resolutionRatio'])
    processedVisium['tissueRotated'] = rotate(processedVisium['tissueResized'], rotation, resize=True)
    processedVisium['tissueHistMatched'] = match_histograms(processedVisium['tissueRotated'], templateData['leftHem'])
    processedVisium['tissueHistMatched'] = processedVisium['tissueHistMatched'] - processedVisium['tissueHistMatched'].min()
    processedVisium['tissuePointsRotated'] = rotateTissuePoints(visiumData, rotation)
    processedVisium['tissuePointsResized'] = processedVisium['tissuePointsRotated'] * processedVisium['resolutionRatio']
    processedVisium['tissuePointsResizedForTransform'] = processedVisium['tissuePointsRotated'] * processedVisium['resolutionRatio']
    processedVisium['tissuePointsResizedForTransform'][:,[0,1]] = processedVisium['tissuePointsResizedForTransform'][:,[1,0]]
    # plt.imshow( processedVisium['tissueRotated'])
    # plt.plot(processedVisium['tissuePointsResized'][:,0],processedVisium['tissuePointsResized'][:,1],marker='.', c='red', alpha=0.2)
    # plt.show()
    
    # no need to keep the dense version in the processedVisium dictionary since the necessary information is in the ordered matrix
    denseMatrix = visiumData["filteredFeatureMatrix"][2]
    denseMatrix = denseMatrix.todense()
    filteredFeatureMatrixString = []
    for bytebarcode in visiumData['filteredFeatureMatrix'][1]:
        filteredFeatureMatrixString.append(bytebarcode.decode())
    processedVisium['filteredFeatureMatrixBarcodeList'] = filteredFeatureMatrixString 

    filteredFeatureMatrixGeneString = []
    for bytegene in visiumData['filteredFeatureMatrix'][0]['name']:
        filteredFeatureMatrixGeneString.append(bytegene.decode())
    processedVisium['filteredFeatureMatrixGeneList'] = filteredFeatureMatrixGeneString
    header=['x','y','z','t','label','comment']
    # csvFormat = []
    rowFormat = []
    with open(f"{os.path.join(outputPath,processedVisium['sampleID'])}_tissuePointsResizeToTemplate.csv", 'w', encoding='UTF8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i in range(len(processedVisium['tissuePointsResizedForTransform'])):
            rowFormat = [processedVisium['tissuePointsResizedForTransform'][i,0]] + [processedVisium['tissuePointsResizedForTransform'][i,1]] + [0] + [0] + [0] + [0]
            writer.writerow(rowFormat)
            # csvFormat.append(rowFormat)
            
    # processedVisium['tissuePointsForTransform'] = np.array(csvFormat)
    
    # this orders the filtered feature matrix so that the columns are in the order of the coordinate list, so barcodes no longer necessary
    filteredFeatureMatrixBarcodeReorder = []
    for actbarcode in processedVisium['tissueSpotBarcodeList']:
        filteredFeatureMatrixBarcodeReorder.append(processedVisium['filteredFeatureMatrixBarcodeList'].index(actbarcode))
    
    processedVisium['filteredFeatureMatrixOrdered'] = denseMatrix[:,filteredFeatureMatrixBarcodeReorder]
    processedVisium['filteredFeatureMatrixLog2'] = np.log2((processedVisium['filteredFeatureMatrixOrdered'] + 1))
    sp_sparse.save_npz(f"{os.path.join(outputPath,processedVisium['sampleID'])}_tissuePointOrderedFeatureMatrix.npz", sp_sparse.csc_matrix(processedVisium['filteredFeatureMatrixOrdered']))
    # np.savetxt("f{os.path.join(outputPath,processedVisium['sampleID'])}_tissuePointOrderedFeatureMatrix.csv", processedVisium['filteredFeatureMatrixOrdered'], delimiter=",")
            
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
    type_of_transform='SyNBoldAff', grad_step=0.1, reg_iterations=(120, 100,80,60,40,20,0), \
    syn_sampling=2.5, flow_sigma=1.5,syn_metric='meansquares', outprefix=os.path.join(processedVisium['derivativesPath'],f"{processedVisium['sampleID']}_xfm"))
    registeredData['antsOutput'] = synXfm
    registeredData['sampleID'] = processedVisium['sampleID']
    registeredData['derivativesPath'] = processedVisium['derivativesPath']
    # apply syn transform to tissue spot coordinates
    applyTransformStr = f"antsApplyTransformsToPoints -d 2 -i {os.path.join(processedVisium['derivativesPath'],processedVisium['sampleID'])}_tissuePointsResizeToTemplate.csv -o {os.path.join(processedVisium['derivativesPath'],processedVisium['sampleID'])}_tissuePointsResizeToTemplateTransformApplied.csv -t [ {os.path.join(processedVisium['derivativesPath'],processedVisium['sampleID'])}_xfm0GenericAffine.mat,1] -t [{os.path.join(processedVisium['derivativesPath'],processedVisium['sampleID'])}_xfm1InverseWarp.nii.gz]"
    pid = os.system(applyTransformStr)
    
    if pid:
        os.wait()
        print("Applying transformation to spots")
    else:
        print("Finished transforming spots!")
    
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

    # plt.imshow(registeredData['visiumTransformed'])
    # plt.scatter(registeredData['transformedTissuePositionList'][0:,0],registeredData['transformedTissuePositionList'][0:,1], marker='.', c='red', alpha=0.3)
    # plt.show()
    
    # plt.imshow(registeredData['visiumTransformed'],cmap='gray')
    # plt.imshow(templateData['leftHem'], alpha=0.3)
    # plt.title(processedVisium['sampleID'])
    # plt.show()
        
    transformedTissuePositionListMask = np.logical_and(registeredData['transformedTissuePositionList'] > 0, registeredData['transformedTissuePositionList'] < registeredData['visiumTransformed'].shape[0])
    transformedTissuePositionListFinal = []
    # filteredFeatureMatrixBinaryMask = []
    # filteredFeatureMatrixMasked = np.zeros(processedVisium['filteredFeatureMatrixOrdered'][:,0].shape)
    filteredFeatureMatrixMaskedIdx = []
    for i, masked in enumerate(transformedTissuePositionListMask):
        if masked.all() == True:
            filteredFeatureMatrixMaskedIdx.append(i)
            # filteredFeatureMatrixBinaryMask.append(1)
            transformedTissuePositionListFinal.append(registeredData['transformedTissuePositionList'][i])
            # filteredFeatureMatrixMasked = np.append(filteredFeatureMatrixMasked, processedVisium['filteredFeatureMatrixOrdered'][:,i],axis=1)
        # else:
            # filteredFeatureMatrixBinaryMask.append(0)
    registeredData['maskedTissuePositionList'] = np.array(transformedTissuePositionListFinal, dtype=float)

    # registeredData['filteredFeatureMatrixMasked'] = np.delete(filteredFeatureMatrixMasked, 0,1)
    registeredData['filteredFeatureMatrixMasked'] = processedVisium['filteredFeatureMatrixLog2'][:,filteredFeatureMatrixMaskedIdx]
    # write re-ordered filtered feature matrix csv to match tissue spot order
    # csvFormat = []
    # rowFormat = []
    # with open(f"{os.path.join(processedVisium['derivativesPath'],processedVisium['sampleID'])}_tissuePointOrderedFeatureMatrixTemplateMasked.csv", 'w', encoding='UTF8') as f:
    #     writer = csv.writer(f)
    #     for i in range(len(registeredData['filteredFeatureMatrixMasked'])):
    #         rowFormat = registeredData['filteredFeatureMatrixMasked'][i,:]
    #         writer.writerow(rowFormat)
    #         # csvFormat.append(rowFormat)
    sp_sparse.save_npz(f"{os.path.join(processedVisium['derivativesPath'],processedVisium['sampleID'])}_OrderedLog2FeatureMatrixTemplateMasked.npz", sp_sparse.csc_matrix(registeredData['filteredFeatureMatrixMasked']))        
    cv2.imwrite(f"{registeredData['derivativesPath']}/{registeredData['sampleID']}_tissue_registered_to_Allen_slice_{templateData['sliceNumber']}.png",registeredData['visiumTransformed'])
    
    return registeredData

def runANTsInterSampleRegistration(processedVisium, sampleToRegisterTo):
    # convert into ants image type
    registeredData = {}
    templateAntsImage = ants.from_numpy(sampleToRegisterTo['tissueHistMatched'])
    sampleAntsImage = ants.from_numpy(processedVisium['tissueHistMatched'])
    # mattes seems to be most conservative syn_metric
    synXfm = ants.registration(fixed=templateAntsImage, moving=sampleAntsImage, \
    type_of_transform='SyNBoldAff', grad_step=0.1, reg_iterations=(140,120,100,80,60,40,20,0), \
    syn_sampling=1.5, flow_sigma=1.5, syn_metric='meansquares', outprefix=os.path.join(processedVisium['derivativesPath'],f"{processedVisium['sampleID']}_to_{sampleToRegisterTo['sampleID']}_xfm"))
    registeredData['antsOutput'] = synXfm
    registeredData['sampleID'] = processedVisium['sampleID']
    registeredData['derivativesPath'] = processedVisium['derivativesPath']
    # ants.plot(templateAntsImage, overlay=synXfm["warpedmovout"])
    # apply syn transform to tissue spot coordinates
    # first line creates a csv file, second line uses that csv as input for antsApplyTransformsToPoints
    # np.savetxt(f"{os.path.join(processedVisium['derivativesPath'],processedVisium['sampleID'])}_tissuePointsResize_to_{sampleToRegisterTo['sampleID']}.csv",processedVisium['tissuePointsForTransform'], delimiter=',', header="x,y,z,t,label,comment")
    applyTransformStr = f"antsApplyTransformsToPoints -d 2 -i {os.path.join(processedVisium['derivativesPath'],processedVisium['sampleID'])}_tissuePointsResizeToTemplate.csv -o {os.path.join(processedVisium['derivativesPath'],processedVisium['sampleID'])}_tissuePointsResize_to_{sampleToRegisterTo['sampleID']}TransformApplied.csv -t [ {os.path.join(processedVisium['derivativesPath'],processedVisium['sampleID'])}_to_{sampleToRegisterTo['sampleID']}_xfm0GenericAffine.mat,1] -t {os.path.join(processedVisium['derivativesPath'],processedVisium['sampleID'])}_to_{sampleToRegisterTo['sampleID']}_xfm1InverseWarp.nii.gz"
    pid = os.system(applyTransformStr)
    
    if pid:
        os.wait()
        print("Applying transformation to spots")
    else:
        print("Finished transforming spots!")
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
    registeredData['tissueSpotBarcodeList'] = processedVisium['tissueSpotBarcodeList']
    registeredData['filteredFeatureMatrixLog2'] = processedVisium['filteredFeatureMatrixLog2']
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
    # filteredFeatureMatrixBinaryMask = []
    # transformedBarcodesFinal = []
    filteredFeatureMatrixMaskedIdx = []
    # filteredFeatureMatrixMasked = np.zeros(registeredVisium['filteredFeatureMatrixOrdered'][:,0].shape)
    for i, masked in enumerate(transformedTissuePositionListMask):
        if masked.all() == True:
            filteredFeatureMatrixMaskedIdx.append(i)
            transformedTissuePositionListFinal.append(templateRegisteredData['transformedTissuePositionList'][i])
            # filteredFeatureMatrixMasked = np.append(filteredFeatureMatrixMasked, registeredVisium['filteredFeatureMatrixOrdered'][:,i],axis=1)
    templateRegisteredData['maskedTissuePositionList'] = np.array(transformedTissuePositionListFinal, dtype=float)
    templateRegisteredData['filteredFeatureMatrixMasked'] = registeredVisium['filteredFeatureMatrixLog2'][:,filteredFeatureMatrixMaskedIdx]
    #sp_sparse.save_npz(f"{os.path.join(registeredVisium['derivativesPath'],registeredVisium['sampleID'])}_OrderedLog2FeatureMatrixAllenTemplateMasked.npz", sp_sparse.csc_matrix(templateRegisteredData['filteredFeatureMatrixMasked']))
    cv2.imwrite(f"{os.path.join(registeredVisium['derivativesPath'],registeredVisium['sampleID'])}_registered_to_{bestSampleRegisteredToTemplate['sampleID']}_to_Allen.png",templateRegisteredData['visiumTransformed'])

    return templateRegisteredData

#%% import sample list, location, and degrees of rotation from participants.tsv
#sampleList contains sample ids, templateList contains template slices and degrees of rotation to match
bestTemplateSlice = 70
template = chooseTemplateSlice(bestTemplateSlice)

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

processedSamples = {}

for actSample in range(len(truncExperiment['sample-id'])):
    sample = importVisiumData(os.path.join(rawdata, truncExperiment['sample-id'][actSample]))
    sampleProcessed = processVisiumData(sample, template, truncExperiment['rotation'][actSample])
    processedSamples[actSample] = sampleProcessed

#%% register to "best" sample
# in this case just the best looking slice
bestSample = processedSamples[4]

bestSampleToTemplate = runANTsToAllenRegistration(bestSample, template)

experimentalResults = {}
for actSample in range(len(processedSamples)):
    sampleRegistered = runANTsInterSampleRegistration(processedSamples[actSample], bestSample)
    experimentalResults[actSample] = sampleRegistered


#%%##########################################
# CHECK FOR ACCURACY OF ABOVE REGISTRATIONS #
#############################################

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

# now to remove non-tissue spots
roundedTemplateSpots = np.array(templateSpots.round(), dtype=int)

inTissueTemplateSpots = []
for row in range(len(roundedTemplateSpots)):
    # 15 in the following is to erode around the edge of the brain
    # if template['leftHem'][roundedTemplateSpots[row,1],roundedTemplateSpots[row,0]] > 15:
    #     inTissueTemplateSpots.append(templateSpots[row])
        
    if bestSampleToTemplate['visiumTransformed'][roundedTemplateSpots[row,1],roundedTemplateSpots[row,0]] > 0:
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
        spotCdist = sp_spatial.distance.cdist(templateRegisteredSpots, np.array(actSpot).reshape(1,-1), 'euclidean')
        sortedSpotCdist = np.sort(spotCdist, axis=0)
        actSpotCdist = sortedSpotCdist[0:kNN]
        # spotNNIdx gives the index of the top kSpots nearest neighbors for each digital spot
        spotMeanCdist = np.mean(actSpotCdist)
        blankIdx = np.zeros([kNN,1], dtype=int)
        spotNNIdx = []
        for i in actSpotCdist:
            if spotMeanCdist < 30:
                actNNIdx = np.where(spotCdist == i)[0]
                spotNNIdx.append(actNNIdx[:])
            else:
                # should probably change this from 0s to something like -1
                spotNNIdx = blankIdx
            
        allMeanCdists.append(spotMeanCdist)
        allSpotNN.append(np.array(spotNNIdx))
        
    allSpotNN = np.squeeze(np.array(allSpotNN))
    # should be able to add threshold that removes any spots with a mean cdist > some value
    return allSpotNN, allMeanCdists

#%% this line needs to be incorporated into one of the functions, so only run once

kSpots = 7
nDigitalSpots = len(inTissueTemplateSpots)
nTotalSamples = len(allSamplesToAllen)
for i, regSample in enumerate(allSamplesToAllen):
        
    # removes any spots with fewer than 5000 total gene counts
    countsPerSpot = np.sum(allSamplesToAllen[i]['filteredFeatureMatrixMasked'],axis=0)
    spotMask = countsPerSpot > 5000
    allSamplesToAllen[i]['filteredFeatureMatrixMasked'] = allSamplesToAllen[i]['filteredFeatureMatrixMasked'][:,np.squeeze(np.array(spotMask))]
    allSamplesToAllen[i]['maskedTissuePositionList'] = allSamplesToAllen[i]['maskedTissuePositionList'][np.squeeze(np.array(spotMask)),:]
    # remove genes with no counts
    countsPerGene = np.count_nonzero(np.array(allSamplesToAllen[i]['filteredFeatureMatrixMasked']),axis=1, keepdims=True)
    geneMask = countsPerGene > 30
    # np.count_nonzero(allSamplesToAllen[i]['filteredFeatureMatrixMasked'][geneIndex,:])
    allSamplesToAllen[i]['filteredFeatureMatrixMasked'] = allSamplesToAllen[i]['filteredFeatureMatrixMasked'][np.squeeze(np.array(geneMask)),:]
    geneMaskedGeneList = np.array(allSamplesToAllen[i]['filteredFeatureMatrixGeneList'])[np.squeeze(np.array(geneMask))]
    actNN, actCDist = findDigitalNearestNeighbors(inTissueTemplateSpots, allSamplesToAllen[i]['maskedTissuePositionList'], kSpots)
    allSamplesToAllen[i]['geneListMasked'] = np.ndarray.tolist(geneMaskedGeneList)
    allSamplesToAllen[i]['digitalSpotNearestNeighbors'] = np.asarray(actNN, dtype=int)
    allSamplesToAllen[i]['zScoredFeatureMatrixMasked'] = (allSamplesToAllen[i]['filteredFeatureMatrixMasked'] - np.mean(allSamplesToAllen[i]['filteredFeatureMatrixMasked'],axis=1)) / np.std(allSamplesToAllen[i]['filteredFeatureMatrixMasked'],axis=1)
    
#%% compare gene lists and find genes present in all samples
# create list of genes present to all slices
allSampleGeneList = allSamplesToAllen[0]['geneListMasked']
for i, regSample in enumerate(allSamplesToAllen):
    if i == 0:
        continue
    allSampleGeneList = set(allSampleGeneList) & set(allSamplesToAllen[i]['geneListMasked'])
        
#%% read gene list from txt file
def loadGeneListFromTxt(locOfTextFile):
    geneListFromTxt = []
    with open(locOfTextFile) as f:
        for gene in f:
            geneListFromTxt.append(gene.strip('\n'))
    return geneListFromTxt
#### everything from here on out including experimental or control in variables needs to be reworked into functions
#%% can now use this gene list to loop over expressed genes 
import scipy
# from statsmodels.stats.multitest import multipletests
import matplotlib.colors as mcolors
import time


# 'Arc','Egr1','Lars2','Ccl4'
testGeneList = ['Arc','Egr1']
caudoputamenGeneList = ['Adora2a','Drd2','Pde10a','Drd1','Scn4b','Gpr6','Ido1','Adcy5','Rasd2','Meis2','Lars2','Ccl4']
allocortexGeneList = ['Nptxr','Lmo3','Slc30a3','Syn2','Snca','Ccn3','Bmp3','Olfm1','Ldha','Tafa2']
fibertractsGeneList = ['Plp1','Mag','Opalin','Cnp','Trf','Cldn11','Cryab','Mobp','Qdpr','Sept4']
hippocampalregionGeneList = ['Wipf3','Cabp7','Cnih2','Gria1','Ptk2b','Cebpb','Nr3c2','Lct','Arhgef25','Epha7']
hypothalamusGeneList = ['Gpx3','Resp18','AW551984','Minar2','Nap1l5','Gabrq','Pcbd1','Sparc','Vat1','6330403K07Rik']
neocortexGeneList = ['1110008P14Rik','Ccl27a','Mef2c','Tbr1','Cox8a','Snap25','Nrgn','Vxn','Efhd2','Satb2']
striatumlikeGeneList = ['Hap1','Scn5a','Pnck','Ahi1','Snhg11','Galnt16','Pnmal2','Baiap3','Ly6h','Meg3']
thalamusGeneList = ['Plekhg1','Tcf7l2','Ntng1','Ramp3','Rora','Patj','Rgs16','Nsmf','Ptpn4','Rab37']
testGeneList = testGeneList + caudoputamenGeneList + allocortexGeneList + fibertractsGeneList + hippocampalregionGeneList + hypothalamusGeneList + neocortexGeneList + striatumlikeGeneList + thalamusGeneList
sigGenes = []

listOfSigGenes220812 = ['Tdp1','Oxsm','Homer1','Katna1','Slc52a3','Btaf1','Aff3','Gm10561','Mtrf1l','Ergic2','Lims1','Gpr3','Serinc2','Arc','Vgf','Trib1','Itpkc','Ier5','Cep57l1','Dlx5','Ccdc151','Tfr2','Colgalt2','Camk1g','Mir124a-1hg','Gm27003','Tnfrsf25','Npas4','Rgs6','Gm21887','Synj2']

start_time = time.time()

nSampleExperimental = sum(truncExperiment['experimental-group'])
nSampleControl = len(truncExperiment['experimental-group']) - nSampleExperimental

alphaSidak = 1 - np.power((1 - 0.05),(1/nDigitalSpots))
# alphaSidak = 5e-8
# list(allSampleGeneList)[0:1000]
for nOfGenesChecked,actGene in enumerate(allSampleGeneList):
    digitalSamplesControl = np.zeros([nDigitalSpots,(nSampleControl * kSpots)])
    digitalSamplesExperimental = np.zeros([nDigitalSpots,(nSampleExperimental * kSpots)])
    startControl = 0
    stopControl = kSpots
    startExperimental = 0
    stopExperimental = kSpots
    nTestedSamples = 0
    nControls = 0
    nExperimentals = 0
    for actSample in range(nTotalSamples):
        try:
            geneIndex = allSamplesToAllen[actSample]['geneListMasked'].index(actGene)
        except(ValueError):
            print(f'{actGene} not in dataset')
            continue

        geneCount = np.zeros([nDigitalSpots,kSpots])
        for spots in enumerate(allSamplesToAllen[actSample]['digitalSpotNearestNeighbors']):
            if np.all(spots[1] < 0):
                geneCount[spots[0]] = 0
            else:
                geneCount[spots[0]] = allSamplesToAllen[actSample]['filteredFeatureMatrixMasked'][geneIndex,np.asarray(spots[1], dtype=int)]
                
        spotCount = np.nanmean(geneCount, axis=1)
        nTestedSamples += 1
        if truncExperiment['experimental-group'][actSample] == 0:
            digitalSamplesControl[:,startControl:stopControl] = geneCount
            startControl += kSpots
            stopControl += kSpots
            nControls += 1
        elif truncExperiment['experimental-group'][actSample] == 1:
            digitalSamplesExperimental[:,startExperimental:stopExperimental] = geneCount
            startExperimental += kSpots
            stopExperimental += kSpots
            nExperimentals += 1
            
        else:
            continue
    
    
    digitalSamplesControl = np.array(digitalSamplesControl, dtype=float).squeeze()
    digitalSamplesExperimental = np.array(digitalSamplesExperimental, dtype=float).squeeze()
    ############################
    # this will check that at least a certain number of spots show expression for the gene of interest #
    ##################
    checkControlSamples = np.count_nonzero(digitalSamplesControl,axis=1)
    checkExperimentalSamples = np.count_nonzero(digitalSamplesExperimental,axis=1)
    checkAllSamples = checkControlSamples & checkExperimentalSamples > 20
    if sum(checkAllSamples) < 20:
        continue
    else:
        testControlSamples = digitalSamplesControl[checkAllSamples,:] 
        testExperimentalSamples = digitalSamplesExperimental[checkAllSamples,:]
        testSpotCoordinates = inTissueTemplateSpots[checkAllSamples,:]
        maskedDigitalSamplesControl = np.zeros(digitalSamplesControl.shape)
        maskedDigitalSamplesExperimental = np.zeros(digitalSamplesExperimental.shape)
        maskedDigitalSamplesControl[checkAllSamples,:] = digitalSamplesControl[checkAllSamples,:]
        maskedDigitalSamplesExperimental[checkAllSamples,:] = digitalSamplesExperimental[checkAllSamples,:]
        maskedTtests = []
        allTstats = np.zeros(nDigitalSpots)
        allPvals = []
        actTtest = scipy.stats.ttest_ind(digitalSamplesExperimental,digitalSamplesControl, axis=1, nan_policy='propagate')
        actTstats = actTtest[0]
        actPvals = actTtest[1]
        mulCompResults = actPvals < alphaSidak
        # mulCompResults = multipletests(actTtest[1], 0.05, method='bonferroni', is_sorted=False)
        # fdrAlpha = mulCompResults[3]
        spotThr = 20 #0.05 * nDigitalSpots
        if sum(mulCompResults) > spotThr:
            sigGenes.append(actGene)
            maskedDigitalCoordinates = inTissueTemplateSpots[np.array(mulCompResults)]
            maskedTstats = actTtest[0][mulCompResults]
            maskedDigitalCoordinates = np.array(maskedDigitalCoordinates)
            medianDigitalControl = np.nanmedian(digitalSamplesControl,axis=1)
            medianDigitalExperimental = np.nanmedian(digitalSamplesExperimental,axis=1)
            meanDigitalControl = scipy.stats.mode(digitalSamplesControl,axis=1)
            meanDigitalExperimental = scipy.stats.mode(digitalSamplesExperimental,axis=1)
            finiteMin = np.nanmin(actTtest[0])
            finiteMax = np.nanmax(actTtest[0])
            if finiteMin > 0:
                finiteMin = -1
            elif finiteMax < 0:
                finiteMax = 1
            zeroCenteredCmap = mcolors.TwoSlopeNorm(0,vmin=finiteMin, vmax=finiteMax)
            tTestColormap = zeroCenteredCmap(actTtest[0])
            maxGeneCount = np.nanmax([medianDigitalControl,medianDigitalExperimental])
            # display mean gene count for control group            
            plt.imshow(bestSampleToTemplate['visiumTransformed'],cmap='gray')
            plt.scatter(inTissueTemplateSpots[:,0],inTissueTemplateSpots[:,1], c=np.array(meanDigitalControl[0]), alpha=0.8, vmin=0,vmax=maxGeneCount,plotnonfinite=False,cmap='Reds')
            plt.title(f'Mean gene count for {actGene}, non sleep deprived')
            plt.colorbar()
            plt.savefig(os.path.join(derivatives,f'meanGeneCount{actGene}Control.png'), bbox_inches='tight', dpi=300)
            plt.show()
            # display mean gene count for experimental group
            plt.imshow(bestSampleToTemplate['visiumTransformed'],cmap='gray')
            plt.scatter(inTissueTemplateSpots[:,0],inTissueTemplateSpots[:,1], c=np.array(meanDigitalExperimental[0]), alpha=0.8, vmin=0,vmax=maxGeneCount,plotnonfinite=False,cmap='Reds')
            plt.title(f'Mean gene count for {actGene}, sleep deprived')
            plt.colorbar()
            plt.savefig(os.path.join(derivatives,f'meanGeneCount{actGene}SleepDep.png'), bbox_inches='tight', dpi=300)
            plt.show()
            # display t statistics
            plt.imshow(bestSampleToTemplate['visiumTransformed'],cmap='gray')
            plt.scatter(inTissueTemplateSpots[:,0],inTissueTemplateSpots[:,1], c=np.array(actTtest[0]), cmap='seismic',alpha=0.8,norm=zeroCenteredCmap,plotnonfinite=False)
            plt.title(f't-statistic for {actGene}.')
            plt.colorbar()
            plt.savefig(os.path.join(derivatives,f'tStatGeneCount{actGene}SleepDep.png'), bbox_inches='tight', dpi=300)
            plt.show()

with open(os.path.join(derivatives,'listOfSigSleepDepGenes.csv'), 'w', encoding='UTF8') as f:
    writer = csv.writer(f)
    for i in sigGenes:
        writer.writerow([i])
print("--- %s seconds ---" % (time.time() - start_time))



# #%% extract atlas information
# from allensdk.core.reference_space_cache import ReferenceSpaceCache
# reference_space_key = 'annotation/ccf_2017'
# resolution = 10
# rspc = ReferenceSpaceCache(resolution, reference_space_key, manifest='manifest.json')
# rsp = rspc.get_reference_space()
# #%%
# # ID 1 is the adult mouse structure graph
# tree = rspc.get_structure_tree(structure_graph_id=1) 
# regionList = tree.get_name_map()
# hippocampus = tree.get_structures_by_name(['Hippocampal formation'])

# hippocampal3dMask = rsp.make_structure_mask([hippocampus[0]['id']])
# hippocampalMask = hippocampal3dMask[700,:,570:]

# rhinalFissure = tree.get_structures_by_name(['rhinal fissure'])
# rhinalFissure3dMask = rsp.make_structure_mask([rhinalFissure[0]['id']])
# rhinalFissureMask = rhinalFissure3dMask[700,:,570:]

# plt.imshow(hippocampalMask)
# plt.show()
# bestTemplateSlice10 = bestTemplateSlice * 10
# plt.imshow(bestSampleToTemplate['visiumTransformed'])
# plt.imshow(hippocampalMask, alpha=0.3)
# plt.show()

# plt.imshow(rhinalFissureMask)
# plt.show()
# bestTemplateSlice10 = bestTemplateSlice * 10
# plt.imshow(bestSampleToTemplate['visiumTransformed'])
# plt.imshow(hippocampalMask, alpha=0.3)
# plt.show()
# #%% create digital hippocampal spots

# hippocampalDigitalTemplateSpots = []
# # spotIdx gives a list of spots within the hippocampal formation
# spotIdx = []
# for row in range(len(roundedTemplateSpots)):
#     # 15 in the following is just to erode around the edge of the brain
#     if hippocampalMask[roundedTemplateSpots[row,1],roundedTemplateSpots[row,0]] == 1:
#         hippocampalDigitalTemplateSpots.append(templateSpots[row])
#         spotIdx.append(row)
        
# hippocampalDigitalTemplateSpots = np.array(hippocampalDigitalTemplateSpots)
# plt.imshow(template['leftHem'])
# plt.scatter(hippocampalDigitalTemplateSpots[:,0],hippocampalDigitalTemplateSpots[:,1], alpha=0.3)
# plt.show()

# #%% check for significant genes within a region/list of regions
# # needs to take an input of regions defined in allen ccf
# # should be able to use original spots when searching roi
# # run through entire gene list looking for a change in expression between conditions
# nSigGenes = 0
# partialGeneList = list(allSampleGeneList)[0:500]
# for nOfGenesChecked,actGene in enumerate(partialGeneList):
#     digitalSamplesControl = []
#     digitalSamplesExperimental = []
#     nTestedSamples = 0
#     nControls = 0
#     nExperimentals = 0
#     for actSample in range(nTotalSamples):
#         try:
#             geneIndex = allSamplesToAllen[actSample]['geneListMasked'].index(actGene)
#             spotCheck = np.count_nonzero(allSamplesToAllen[actSample]['filteredFeatureMatrixMasked'][spotIdx,:])
#             # need to do better than "no expression" because that eliminates the fact that the amount is actually 0, not untested
            
#             # use binary mask to remove any tissue spots with no expression
#             if spotCheck < 15:
#                 continue
#             actNN, actCDist = findDigitalNearestNeighbors(hippocampalDigitalTemplateSpots, allSamplesToAllen[actSample]['maskedTissuePositionList'], kSpots)
#             geneCount = np.zeros([hippocampalDigitalTemplateSpots.shape[0],kSpots])
#             for spots in enumerate(actNN):
#                 if ~np.all(spots[1]):
#                     geneCount[spots[0]] = 0
#                 else:
#                     geneCount[spots[0]] = allSamplesToAllen[actSample]['filteredFeatureMatrixMasked'][geneIndex,actNN[spots[0]]]
                    
#             geneCount = geneCount.reshape([-1])
#             nTestedSamples += 1
#             if truncExperiment['experimental-group'][actSample] == 0:
#                 digitalSamplesControl.append(geneCount)
#                 nControls += 1
#             elif truncExperiment['experimental-group'][actSample] == 1:
#                 digitalSamplesExperimental.append(geneCount)
#                 nExperimentals += 1
                
#             else:
#                 continue
            
#         except:
#             continue
        
#     if spotCheck < 15:
#         continue    
#     digitalSamplesControl = np.array(digitalSamplesControl)
#     digitalSamplesExperimental = np.array(digitalSamplesExperimental)
#     maskedTtests = []
#     allTstats = []
#     allPvals = []
#     if ~digitalSamplesControl.any() or ~digitalSamplesExperimental.any():
#         continue
#     else:    
#         actTtest = scipy.stats.ttest_ind(digitalSamplesExperimental,digitalSamplesControl, nan_policy='omit', axis=None)
#         maskedDigitalCoordinates = []
#         maskedMeanDigitalControls = []
#         maskedMeanDigitalExperimentals = []
#         meanDigitalControlHippocampalFormation = np.mean(digitalSamplesControl)
#         meanDigitalExperimentalHippocampalFormation = np.mean(digitalSamplesExperimental)
#         if actTtest[1] <= 0.05:
#             nSigGenes += 1
#             maxGeneCount = np.max([meanDigitalControls,meanDigitalExperimentals])
#             tStatColor = np.full(hippocampalDigitalTemplateSpots.shape[0],actTtest[0])
#             plt.imshow(bestSampleToTemplate['visiumTransformed'])
#             if actTtest[0] < 0:
#                 plt.scatter(hippocampalDigitalTemplateSpots[:,0],hippocampalDigitalTemplateSpots[:,1], c='blue', alpha=0.8,plotnonfinite=False)
#             else:
#                 plt.scatter(hippocampalDigitalTemplateSpots[:,0],hippocampalDigitalTemplateSpots[:,1], c='red', alpha=0.8,plotnonfinite=False)

#             # plt.scatter(hippocampalDigitalTemplateSpots[:,0],hippocampalDigitalTemplateSpots[:,1], c=tStatColor, alpha=0.8, vmin=0,vmax=maxGeneCount,plotnonfinite=False)
#             # plt.title(f'Mean gene count for {actGene}, control')
#             plt.title(f'{actGene}, non sleep deprived, hippocampal expression p <= 0.05')
#             plt.colorbar()
#             plt.show()
            
#         else:
#             continue
        
        
        
# print("--- %s seconds ---" % (time.time() - start_time))
