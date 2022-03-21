#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 09:49:40 2022

@author: zjpeters
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
from scipy.spatial.distance import pdist, squareform

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
    plt.imshow(visiumData['imageData'])
    plt.show()
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
    csvPad = np.zeros([processedVisium['tissuePointsResizedForTransform'].shape[0],4])
    processedVisium['tissuePointsForTransform'] = np.append(processedVisium['tissuePointsResizedForTransform'], csvPad, 1)
    plt.imshow( processedVisium['tissueRotated'])
    plt.plot(processedVisium['tissuePointsResized'][:,0],processedVisium['tissuePointsResized'][:,1],marker='.', c='red', alpha=0.2)
    plt.show()
    processedVisium['filteredFeatureMatrixDense'] = visiumData["filteredFeatureMatrix"][2]
    processedVisium['filteredFeatureMatrixDense'] = processedVisium['filteredFeatureMatrixDense'].todense()
    return processedVisium

# will have to add right left hemisphere choice, eventually potentially sagittal etc
def runANTsRegistration(processedVisium, templateData):
    # convert into ants image type
    registeredData = {}
    templateAntsImage = ants.from_numpy(templateData['leftHem'])
    sampleAntsImage = ants.from_numpy(processedVisium['tissueHistMatched'])
    synXfm = ants.registration(fixed=templateAntsImage, moving=sampleAntsImage, \
    type_of_transform='SyNAggro', grad_step=0.1, reg_iterations=(100,80,60,40,20,0), \
    syn_sampling=2, flow_sigma=2, syn_metric='mattes', outprefix=os.path.join(processedVisium['derivativesPath'],f"{processedVisium['sampleID']}_xfm"))
    registeredData['antsOutput'] = synXfm
    ants.plot(templateAntsImage, overlay=synXfm["warpedmovout"])
    # apply syn transform to tissue spot coordinates
    # first line creates a csv file, second line uses that csv as input for antsApplyTransformsToPoints
    np.savetxt(f"{os.path.join(processedVisium['derivativesPath'],processedVisium['sampleID'])}_tissuePointsResizeToTemplate.csv",processedVisium['tissuePointsForTransform'], delimiter=',', header="x,y,z,t,label,comment")
    os.system(f"antsApplyTransformsToPoints -d 2 -i {os.path.join(processedVisium['derivativesPath'],processedVisium['sampleID'])}_tissuePointsResizeToTemplate.csv -o {os.path.join(processedVisium['derivativesPath'],processedVisium['sampleID'])}_tissuePointsResizeToTemplateTransformApplied.csv -t [ {os.path.join(processedVisium['derivativesPath'],processedVisium['sampleID'])}_xfm0GenericAffine.mat,1] -t {os.path.join(processedVisium['derivativesPath'],processedVisium['sampleID'])}_xfm1InverseWarp.nii.gz")
    
    transformedTissuePositionList = []
    with open(os.path.join(f"{os.path.join(processedVisium['derivativesPath'],processedVisium['sampleID'])}_tissuePointsResizeToTemplateTransformApplied.csv"), newline='') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',')
            next(csvreader)
            for row in csvreader:
                transformedTissuePositionList.append(row)
                
    registeredData['visiumTransformed'] = synXfm["warpedmovout"].numpy()

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
    
    transformedTissuePositionListFinal = [];
    transformedBarcodesFinal = []
    for i, masked in enumerate(transformedTissuePositionListMask):
        if masked.all() == True:
            transformedTissuePositionListFinal.append(registeredData['transformedTissuePositionList'][i])
            transformedBarcodesFinal.append(processedVisium["tissueSpotBarcodeList"][i])
    
    registeredData['maskedTissuePositionList'] = np.array(transformedTissuePositionListFinal, dtype=float)
    registeredData['maskedBarcodes'] = transformedBarcodesFinal
    return registeredData
    
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
              'rotation': templateList[:,1]}

truncExperiment = {'sample-id': np.asarray(sampleList)[imageList],
                   'template-slice': templateList[imageList,0],
                   'rotation': templateList[imageList,1]}

#%% import sample data
# degreesToRotate = 180
# allenSlice = 69
# actSample = "../rawdata/sleepDepBothBatches/sample-09"
experimentalResults = {}
# working on below bit
# experimentalResults = dict.fromkeys(['sample-id','antsOutput'])
for actSample in range(len(experiment['sample-id'])):
    sample = importVisiumData(os.path.join(rawdata, experiment['sample-id'][actSample]))
    template = chooseTemplateSlice(experiment['template-slice'][actSample])
    sampleProcessed = processVisiumData(sample, template, experiment['rotation'][actSample])
    sampleRegistered = runANTsRegistration(sampleProcessed, template)
    experimentalResults[actSample] = sampleRegistered

#%%##########################################
# CHECK FOR ACCURACY OF ABOVE REGISTRATIONS #
#############################################

#%% calculate pairwise distance for each points in a sample
# kNN here is how many nearest neighbors we want to calculate
kNN = 12

pairwiseSquareMatrix = {}
pairwiseNearestNeighbors = {}
nearestNeighborEdges = {}
####
# need to adjust/build edges, since right now two nearest neighbors with the
# same distance is causing a crash because of multiple indices
#### ^ was a euclidean metric issue, changing metric in pdist fixes
for actSample in range(len(experimentalResults)):    
    print(experiment['sample-id'][actSample])
    samplePDist = []
    samplePDist = pdist(experimentalResults[actSample]['maskedTissuePositionList'], metric='cosine')
    samplePDistSM = []
    samplePDistSM = squareform(samplePDist)
    pairwiseSquareMatrix[actSample] = samplePDistSM
    samplePDistSMSorted = []
    samplePDistSMSorted = np.sort(samplePDistSM, axis=1)
    # below contains kNN distances for each in tissue spot based on post alignment distance
    samplePDistNN = []
    samplePDistNN = samplePDistSMSorted[:,1:kNN+1]
    samplePDistEdges = []
    # output of samplekNN should contain the barcode indices of all of the nearest neighbors
    samplekNN = np.zeros(samplePDistNN.shape)
    for i, row in enumerate(samplePDistSM):
        for sigK in range(kNN):
            samplekNN[i,sigK] = np.argwhere(row == samplePDistNN[i,sigK])
            samplePDistEdges.append([i,np.argwhere(row == samplePDistNN[i,sigK])]) 
            # samplePDistEdges[1,i] = 
            
    pairwiseNearestNeighbors[actSample] = samplekNN
    nearestNeighborEdges[actSample] = samplePDistEdges
#%% take nearest neighbor lists and turn into list of coordinate edges i.e. [I,J] 

sampleEdges = []
for actSample in pairwiseNearestNeighbors:
    for i, row in enumerate(actSample):
        sampleEdges.append(())


#%% next steps towards clustering data, though this could realistically be replaced by something like BayesSpace
# from sklearn.cluster import AffinityPropagation
# from sklearn import metrics

# from scipy import spatial

#%% run cosine similarity on kNN spots
V = []
v = np.zeros([samplekNN.shape[0],samplekNN.shape[1]])
for i in range(samplekNN.shape[0]):
    for k, j in enumerate(samplekNN[i]):
        V = cosine(sampleProcessed['filteredFeatureMatrixDense'][:,i], sampleProcessed['filteredFeatureMatrixDense'][:,j])
        v[i,k] = 1 - V



#%% create 3d image from selected sample runs
# # list of closer images
# nearbyImageList = [4,5,6,7]
# allCoordinates = np.zeros([1,2])
# for i in imageList:
#     np.append(allCoordinates, experimentalResults[i]['maskedTissuePositionList'])