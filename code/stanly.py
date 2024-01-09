#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STANLY is a code base built in order to process and analyze various types of 
spatial transcriptomic data. This includes masking using images, spots, cells, etc.
This project has been created and maintained by Zeru Peterson at the University of Iowa

@author: Zeru Peterson, zeru-peterson@uiowa.edu https://research-git.uiowa.edu/zjpeters/STANLY
"""
__version__ = '0.0.2'
__author__ = 'Zeru Peterson'
# data being read in includes: json, h5, csv, nrrd, jpg, tiff, and svg
import os
from skimage import io, filters, color, feature, morphology
from skimage.transform import rescale, rotate
from skimage.exposure import match_histograms
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import json
import csv
import cv2
from glob import glob
import ants
import scipy
import scipy.spatial as sp_spatial
import scipy.sparse as sp_sparse
import collections
import tables
# import time
import itk
import pandas as pd
from matplotlib.widgets import LassoSelector
from matplotlib.path import Path
from PIL import ImageColor
# grab location of the CCF reference data
codePath = os.path.realpath(os.path.dirname(__file__))
dataPath = os.path.join(codePath, 'data')

""" 
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

""" the general workflow should go as follows:
    1. import spatial transcriptomic/genomic data
    2. import relevant atlas images and annotations from allen atlas
    3. prepare data for registration into Common Coordinate Framework (CCF)
    4. use SyN registration from ANTs to register histological image to allen image
    5. bring remaining data, such as spot coordinates, into allen space using above transformations
    6. measure for nearest neighbor similarity among spots in new space and create a vector that represents the nearest neighbors from each slice
"""

#%% import functions
def importVisiumData(sampleFolder):
    """
    Imports individual Visium experiment from location

    Parameters
    ----------
    sampleFolder : directory containing Visium data
    Returns
    ----------
    visiumData : dictionary containing Visium data
    """
    visiumData = {}
    if os.path.exists(os.path.join(sampleFolder,"spatial")):
        spatialFolder = os.path.join(sampleFolder,"spatial")
        try:
            os.path.isfile(glob(os.path.join(sampleFolder, '*filtered_feature_bc_matrix.h5'))[0])
            dataFolder = sampleFolder
        except IndexError:
            os.path.isfile(glob(os.path.join(spatialFolder, '*filtered_feature_bc_matrix.h5'))[0])
            dataFolder = spatialFolder
    elif os.path.exists(os.path.join(sampleFolder,"outs","spatial")):
        spatialFolder = os.path.join(sampleFolder,"outs","spatial")
        try:
            os.path.isfile(glob(os.path.join(sampleFolder,"outs", '*filtered_feature_bc_matrix.h5'))[0])
            dataFolder = os.path.join(sampleFolder,"outs")
        except IndexError:
            os.path.isfile(glob(os.path.join(spatialFolder, '*filtered_feature_bc_matrix.h5'))[0])
            dataFolder = spatialFolder
    else:
        print("Something isn't working!")
    
    visiumData['imageData'] = io.imread(os.path.join(spatialFolder,"tissue_hires_image.png"))
    visiumData['imageDataGray'] = 1 - color.rgb2gray(visiumData['imageData'])
    visiumData['sampleID'] = sampleFolder.rsplit(sep='/',maxsplit=1)[-1]
    tissuePositionList = []
    tissueSpotBarcodes = []
    with open(os.path.join(spatialFolder,"tissue_positions_list.csv"), newline='') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',')
        for row in csvreader:
            # checks for in tissue spots
            if row[1] == '1':
                tissueSpotBarcodes.append(row[0])
                tissuePositionList.append(row[1:])
    tissuePositionList = np.array(tissuePositionList, dtype='float32')
    visiumData['tissueSpotBarcodeList'] = tissueSpotBarcodes
    visiumData['tissuePositionList'] = tissuePositionList
    scaleFactorPath = open(os.path.join(spatialFolder,"scalefactors_json.json"))
    visiumData['scaleFactors'] = json.loads(scaleFactorPath.read())
    scaleFactorPath.close()
    geneMatrixPath = glob(os.path.join(dataFolder,"*filtered_feature_bc_matrix.h5"))
    geneMatrix = get_matrix_from_h5(os.path.join(geneMatrixPath[0]))
    visiumData['geneMatrix'] = geneMatrix
    # the ratio of real spot diameter, 55um, by imaged resolution of spot
    visiumData['spotStartingResolution'] = 0.55 / visiumData["scaleFactors"]["spot_diameter_fullres"]
    return visiumData

def importMerfishData(sampleFolder, outputPath):
    sampleData = {}
    if os.path.exists(os.path.join(sampleFolder)):
        if any(glob(os.path.join(sampleFolder, '*cell_by_gene*.csv'))):
            dataFolder = sampleFolder
            sampleData['sampleID'] = sampleFolder.rsplit(sep='/',maxsplit=1)[-1]
        elif any(glob(os.path.join(sampleFolder, 'region_0','*cell_by_gene*.csv'))):
            
            os.path.isfile(glob(os.path.join(sampleFolder,'region_0','*cell_by_gene*.csv'))[0])
            dataFolder = os.path.join(sampleFolder, 'region_0')
            sampleData['sampleID'] = sampleFolder.rsplit(sep='/',maxsplit=1)[-1]
        else:
            print("Something is wrong!")
    else:
        print(f"{sampleFolder} not found!")
    if any(glob(os.path.join(dataFolder,"images","manifest.json"))):
        scaleFactorPath = open(os.path.join(dataFolder,"images","manifest.json"))
        sampleData['scaleFactors'] = json.loads(scaleFactorPath.read())
        scaleFactorPath.close()
    if any(glob(os.path.join(dataFolder,"*mosaic_DAPI_z0.tif"))):
        originalImagePath =  glob(os.path.join(dataFolder,"*mosaic_DAPI_z0.tif"))[0]
    elif any(glob(os.path.join(dataFolder,"images","*mosaic_DAPI_z0.tif"))):
        originalImagePath =  glob(os.path.join(dataFolder,"images","*mosaic_DAPI_z0.tif"))[0]
    else:
        print(f"Can't find tif in {dataFolder} or {dataFolder}/images")
    downsampledImagePath = os.path.splitext(originalImagePath)[0]
    downsampledImagePath = downsampledImagePath + "_downsampled.tif"
    if any(glob(downsampledImagePath)):
        print(f"Loading previously downsampled image from {dataFolder}")
    else:
        print("Downsampling high resolution image to 10 micron resolution")
        downsampleMerfishTiff(originalImagePath, downsampledImagePath, scale=0.01)
    sampleData['imageData'] = io.imread(downsampledImagePath)
    # min-max normalize image data 
    grayImage = (sampleData['imageData'] - np.min(sampleData['imageData']))\
        /(np.max(sampleData['imageData']) - np.min(sampleData['imageData']))
    sampleData['imageDataGray'] = np.array(grayImage, dtype='float32')
    cellMetadataCsv = glob(os.path.join(dataFolder,"*cell_metadata*.csv"))[0]
    tissuePositionList = []
    tissueSpotBarcodes = []    
    with open(cellMetadataCsv, newline='') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',')
        next(csvreader)
        for row in csvreader:
            tissueSpotBarcodes.append(row[0])
            tissuePositionList.append(row[3:5])
    tissuePositionList = np.array(tissuePositionList, dtype='float32')
    sampleData['tissueSpotBarcodeList'] = tissueSpotBarcodes
    sampleData['tissuePositionList'] = tissuePositionList
    geneMatrixPath = glob(os.path.join(dataFolder,"*cell_by_gene*.csv"))[0]
    geneMatrix = []
    with open(geneMatrixPath, newline='') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',')
        next(csvreader)
        for row in csvreader:
            geneMatrix.append(np.array(row[1:], dtype='float32'))
    sampleData['geneMatrix'] = np.array(geneMatrix, dtype='int32')
    cellByGene = pd.read_csv(geneMatrixPath)
    geneList = cellByGene.columns[1:]
    sampleData['geneList'] = list(geneList) 
    return sampleData

#%% template/allen ccf functions
def chooseTemplateSlice(sliceLocation):
    """
    Choose template slice to align data to
    ----------
    Chooses a single slice from the Allen Common Coordinate Framework (CCF) to use in image registration.
    ----------
    Parameters
    sliceLocation : int
        The slice number from the CCF you wish to load
    """
    ccfPath = os.path.join(dataPath,'ccf')
    # checks if allen ccf data has been downloaded already and downloads it if it hasn't
    if not os.path.exists(ccfPath):
        print("Downloading 10 micron resolution ara_nissl nrrd file from the Allen Institute, this may take awhile")
        os.mkdirs(ccfPath)
        from allensdk.core.reference_space_cache import ReferenceSpaceApi
        rsapi = ReferenceSpaceApi()
        rsapi.download_volumetric_data('ara_nissl','ara_nissl_10.nrrd',10, save_file_path=os.path.join(ccfPath, 'ara_nissl_10.nrrd'))
    ara_data = ants.image_read(os.path.join(ccfPath,'ara_nissl_10.nrrd'))
    bestSlice = sliceLocation * 10
    templateSlice = ara_data.slice_image(0,(bestSlice))
    del(ara_data)
    annotation_data = ants.image_read(os.path.join(ccfPath,'annotation_10.nrrd'))
    templateAnnotationSlice = annotation_data.slice_image(0,(bestSlice))
    del(annotation_data)
    templateData = {}
    annotation_id = []
    annotation_name = []
    structure_id_path = []
    color_hex = []
    with open(os.path.join(dataPath,'allen_ccf_annotation.csv'), newline='') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',')
        next(csvreader)
        for row in csvreader:
            # imports the annotation id, and full name
            annotation_id.append(row[3])
            annotation_name.append(row[4])
            structure_id_path.append(row[7])
            color_hex.append(ImageColor.getcolor(f"#{row[1]}", 'RGB'))
    templateData = {'annotationID':annotation_id,
                    'annotationName':annotation_name,
                    'structureIDPath':structure_id_path,
                    'annotationColor':list(np.array(color_hex)/255)}
    # templateData['annotationID'] = annotation_id
    # templateData['annotationName'] = annotation_name
    # templateData['structureIDPath'] = structure_id_path
    # templateData['annotationColor'] = list(np.array(color_hex)/255)
    # templateData['annotationColorLabels'] = dict(zip(templateData['annotationID'], templateData['annotationColor']))
    templateData['sliceNumber'] = sliceLocation
    # uses the 10 micron CCF
    templateData['startingResolution'] = 0.01

    templateSlice = cv2.normalize(templateSlice.numpy(), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    templateData['wholeBrain'] = templateSlice
    templateData['wholeBrainAnnot'] = np.array(templateAnnotationSlice.numpy(), dtype='int32')
    templateData['templateWholeGauss'] = filters.gaussian(templateSlice, sigma=10)
    
    annotX = templateData['wholeBrainAnnot'].shape[0]
    annotY = templateData['wholeBrainAnnot'].shape[1]
    templateAnnotRGB = np.zeros([annotX, annotY, 3])
    templateAnnotUnique = np.unique(templateData['wholeBrainAnnot'])
    templateAnnotRenum = np.zeros(templateData['wholeBrainAnnot'].shape)
    templateAnnotColorRenum = []
    annotationIDRenum = []
    for newNum, origNum in enumerate(templateAnnotUnique):
        templateAnnotRenum[templateData['wholeBrainAnnot'] == origNum] = newNum
        annotationIDRenum = newNum
        colorIdx = np.where(np.array(templateData['annotationID']) == f'{origNum}')[0]
        if colorIdx:
            newColor = templateData['annotationColor'][colorIdx[0]]
            templateAnnotColorRenum.append(newColor)
    # templateData['annotationColor'] = mcolors.ListedColormap(templateAnnotColorRenum, N=len(templateData['annotationID']))
    se = morphology.disk(2)
    templateAnnotRGB = morphology.binary_dilation(feature.canny(templateAnnotRenum), footprint=se)
    # templateData['annotationID'] = annotationIDRenum
    templateData['wholeBrainAnnotEdges']  = templateAnnotRGB
    
    templateData['leftHem'] = templateSlice[:,:570]
    templateData['leftHemAnnot'] = templateData['wholeBrainAnnot'][:,:570]
    templateData['templateLeftGauss'] = templateData['templateWholeGauss'][:,:570]
    templateData['leftHemAnnotEdges'] = templateAnnotRGB[:,:570]
    
    templateData['rightHem'] = templateSlice[:,570:]
    templateData['rightHemAnnot'] = templateData['wholeBrainAnnot'][:,570:]
    templateData['templateRightGauss'] = templateData['templateWholeGauss'][:,570:]
    templateData['rightHemAnnotEdges'] = templateAnnotRGB[:,570:]

    return templateData

#%% processing functions
def rotateTissuePoints(tissuePoints, tissueImage, theta, flip=False):
    """
    Rotate tissue coordinates and image by theta degrees
    ----------
    Parameters
    tissuePoints : list
        List of 2D coordinates to be rotated
    tissueImage : 2D image
    theta : int
        Degrees to rotate tissue points and image
    flip : bool
        Set as true to flip image across L/R axis, default=False
    """
    rad = np.deg2rad(theta)
    # create rotation matrix
    rotMat = np.array([[np.cos(rad),-(np.sin(rad))],\
              [np.sin(rad),np.cos(rad)]])
    origin = np.array([tissueImage.shape[1]/2,tissueImage.shape[0]/2])
    rotImage = rotate(tissueImage, theta, resize=True)
    # rotation matrix for flipping hemisphere
    if flip==True:
        rotImage = rotImage[:,::-1]
        rotMat = np.array([[-np.cos(rad),(np.sin(rad))],\
                  [np.sin(rad),np.cos(rad)]])
    rotOrigin = np.array([rotImage.shape[1]/2, rotImage.shape[0]/2])
    centeredTP = tissuePoints - origin
    tissuePointsRotate = np.matmul(centeredTP, rotMat)
    tissuePointsRotateCenter = tissuePointsRotate + rotOrigin
    return tissuePointsRotateCenter, rotImage
#%% data processing functions
# prepares visium data for registration
def processVisiumData(visiumData, templateData, rotation, outputFolder, log2normalize=True, flip=False):
    """
    Perform processing on Visium data to prepare for registration and analysis
    ----------
    visiumData : output of importVisiumData
    templateData : output of chooseTemplateSlice
    rotation : int
        degrees of rotation to bring sample into roughly CCF space
    outputFolder : directory
        Location for storing output data
    log2normalize : bool
        Tells function whether to log2normalize gene matrix, default=True
    flip : bool
        Set as true to flip image across L/R axis, default=False
    """
    processedVisium = {}
    processedVisium['sampleID'] = visiumData['sampleID']
    outputPath = os.path.join(outputFolder, visiumData['sampleID'])
    processedVisium['sourceType'] = 'visium'
    try:
        file = open(f"{os.path.join(outputPath,processedVisium['sampleID'])}_tissuePointsProcessed.csv", 'r')
        print(f"{processedVisium['sampleID']} has already been processed! Loading data")
        processedVisium = loadProcessedVisiumSample(outputPath)
        return processedVisium
    except IOError:
        print(f"Processing {processedVisium['sampleID']}")
    if not os.path.exists(outputPath):
        os.makedirs(outputPath)
    resolutionRatio = visiumData['spotStartingResolution'] / templateData['startingResolution']
    processedVisium['derivativesPath'] = outputPath
    # processedVisium['tissueSpotBarcodeList'] = visiumData['tissueSpotBarcodeList']
    processedVisium['degreesOfRotation'] = int(rotation)
    # first gaussian is to calculate otsu threshold
    visiumGauss = filters.gaussian(visiumData['imageDataGray'], sigma=20)
    otsuThreshold = filters.threshold_otsu(visiumGauss)
    # changed to > for inv green channel, originally < below
    processedVisium['visiumOtsu'] = visiumGauss > otsuThreshold
    tissue = np.zeros(visiumGauss.shape, dtype='float32')
    tissue[processedVisium['visiumOtsu']==True] = visiumData['imageDataGray'][processedVisium['visiumOtsu']==True]
    # second gaussian is the one used in registration
    visiumGauss = filters.gaussian(visiumData['imageDataGray'], sigma=5)
    tissueGauss = np.zeros(visiumGauss.shape)
    tissueGauss[processedVisium['visiumOtsu']==True] = visiumGauss[processedVisium['visiumOtsu']==True]
    tissueNormalized = cv2.normalize(tissueGauss, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    tissueResized = rescale(tissueNormalized,resolutionRatio)
    # tissuePointsResizeToHighRes = visiumData["tissuePositionList"][0:, 3:] * visiumData["scaleFactors"]["tissue_hires_scalef"]
    tissuePointsResizeToHighRes = visiumData["tissuePositionList"][0:, 3:] * visiumData["scaleFactors"]["tissue_hires_scalef"] * resolutionRatio
    tissuePointsResizeToHighRes[:,[0,1]] = tissuePointsResizeToHighRes[:,[1,0]] 
    if flip==True:
        tissuePointsRotated, tissueRotated = rotateTissuePoints(tissuePointsResizeToHighRes, tissueResized, rotation, flip=True)
    else:
        tissuePointsRotated, tissueRotated = rotateTissuePoints(tissuePointsResizeToHighRes, tissueResized, rotation)
    processedVisium['tissueProcessed'] = match_histograms(tissueRotated, templateData['rightHem'])
    processedVisium['tissueProcessed'] = processedVisium['tissueProcessed'] - processedVisium['tissueProcessed'].min()
    
    tissuePointsResized = tissuePointsRotated    
    geneMatrixGeneList = []
    for geneName in visiumData['geneMatrix'][0]['name']:
        geneMatrixGeneList.append(geneName.decode())

    # this orders the filtered feature matrix so that the columns are in the order of the coordinate list, so barcodes no longer necessary
    geneMatrixBarcodeList = []
    for barcodeID in visiumData['geneMatrix'][1]:
        geneMatrixBarcodeList.append(barcodeID.decode())
        
    tissueSpotBarcodeListSorted = []
    for actbarcode in visiumData['tissueSpotBarcodeList']:
        tissueSpotBarcodeListSorted.append(geneMatrixBarcodeList.index(actbarcode))
    # no need to keep the dense version in the processedVisium dictionary since the necessary information is in the ordered matrix and coordinates
    denseMatrix = visiumData["geneMatrix"][2]
    denseMatrix = denseMatrix.todense().astype('float32')
    orderedDenseMatrix = denseMatrix[:,tissueSpotBarcodeListSorted]
    countsPerSpot = np.sum(orderedDenseMatrix,axis=0)
    countsPerSpotMean = np.mean(countsPerSpot)
    countsPerSpotStD = np.std(countsPerSpot)
    countsPerSpotZscore = (countsPerSpot - countsPerSpotMean) / countsPerSpotStD
    spotMask = countsPerSpot > 5000
    spotMask = np.squeeze(np.array(spotMask))
    countsPerGene = np.count_nonzero(np.array(orderedDenseMatrix),axis=1, keepdims=True)
    geneMask = countsPerGene > 0
    geneMask = np.squeeze(np.array(geneMask))    
    geneMatrixGeneList = np.array(geneMatrixGeneList)
    processedVisium['geneListMasked'] = geneMatrixGeneList[geneMask].tolist()
    orderedDenseMatrixSpotMasked = orderedDenseMatrix[:,spotMask]
    orderedDenseMatrixSpotMasked = orderedDenseMatrixSpotMasked[geneMask,:]
    processedVisium['spotCount'] = orderedDenseMatrixSpotMasked.shape[1]
    print(f"{processedVisium['sampleID']} has {processedVisium['spotCount']} spots")
    processedVisium['processedTissuePositionList'] = tissuePointsResized[spotMask,:]
    # print(spotMask)
    # processedVisium['tissueSpotBarcodeListSorted'] = tissueSpotBarcodeListSorted[spotMask]
    # processedVisium['filteredFeatureMatrixOrdered'] = sp_sparse.csc_matrix(orderedDenseMatrixSpotMasked)
    if log2normalize==True:
        processedVisium['geneMatrixLog2'] = sp_sparse.csc_matrix(np.log2((orderedDenseMatrixSpotMasked + 1)))
        sp_sparse.save_npz(f"{os.path.join(outputPath,processedVisium['sampleID'])}_tissuePointOrderedFeatureMatrixLog2Normalized.npz", processedVisium['geneMatrixLog2'])
        
    else:
        processedVisium['geneMatrix'] = sp_sparse.csc_matrix(orderedDenseMatrixSpotMasked)
        sp_sparse.save_npz(f"{os.path.join(outputPath,processedVisium['sampleID'])}_tissuePointOrderedFeatureMatrix.npz", processedVisium['geneMatrix'])
        
    finiteMin=np.min(countsPerSpotZscore)
    finiteMax=np.max(countsPerSpotZscore)
    zeroCenteredCmap = mcolors.TwoSlopeNorm(0,vmin=finiteMin, vmax=finiteMax)
    # plt.figure()
    # plt.imshow(processedVisium['tissueProcessed'], cmap='gray')
    # plt.scatter(tissuePointsResized[:,0],tissuePointsResized[:,1], c=np.array(countsPerSpotZscore), alpha=0.8, cmap='seismic', norm=zeroCenteredCmap, marker='.')
    # plt.title(f"Z-score of overall gene count per spot for {processedVisium['sampleID']}")
    # plt.colorbar()
    # plt.show()
    processedVisium['resolutionRatio'] = visiumData['spotStartingResolution'] / templateData['startingResolution']
    processedVisium['locProcessedTissuePointsCSV']  = f"{os.path.join(outputPath,processedVisium['sampleID'])}_tissuePointsProcessed.csv"
    # write outputs
    # writes json containing general info and masked gene list
    processedDataDict = {
        "sampleID": visiumData['sampleID'],
        "rotation": int(rotation),
        "resolutionRatio": processedVisium['resolutionRatio'],
        "spotCount": processedVisium['spotCount'],
        "otsuThreshold": float(otsuThreshold),
        "locProcessedTissuePointsCSV": processedVisium['locProcessedTissuePointsCSV'],
        "geneList": processedVisium['geneListMasked']
    }
        # Serializing json
    json_object = json.dumps(processedDataDict, indent=4)
     
    # Writing to sample.json
    with open(f"{processedVisium['derivativesPath']}/{processedVisium['sampleID']}_processing_information.json", "w") as outfile:
        outfile.write(json_object)
    # writes sorted, masked, normalized filtered feature matrix to .npz file
    
    # writes image for masked greyscale tissue, as well as the processed image that will be used in registration
    cv2.imwrite(f"{processedVisium['derivativesPath']}/{processedVisium['sampleID']}_tissue.png",255*tissue)
    cv2.imwrite(f"{processedVisium['derivativesPath']}/{processedVisium['sampleID']}_tissueProcessed.png",processedVisium['tissueProcessed'])

    # the x and y are swapped between ants and numpy, but this is so far dealt with within the code   
    sortedBarcodeList = np.array(visiumData['tissueSpotBarcodeList'])[spotMask]
    # print(sortedBarcodeList)
    pts = {'x': processedVisium['processedTissuePositionList'][:,1], 'y': processedVisium['processedTissuePositionList'][:,0],'barcode':  sortedBarcodeList}
    pts = pd.DataFrame(pts)
    pts.to_csv(processedVisium['locProcessedTissuePointsCSV'], index=False)
    
    return processedVisium

def processMerfishData(sampleData, templateData, rotation, outputFolder, log2normalize=True):
    processedData = {}
    # the sampleID might have issues on non unix given the slash direction, might need to fix
    processedData['sampleID'] = sampleData['sampleID']
    processedData['sourceType'] = 'merfish'
    processedData['tissuePositionList'] = sampleData['tissuePositionList']
    processedData['geneMatrix'] = np.transpose(sampleData['geneMatrix'])
    processedData['geneList'] = sampleData['geneList']
    processedData['degreeesOfRotation'] = rotation
    outputPath = os.path.join(outputFolder, sampleData['sampleID'])
    try:
        file = open(f"{os.path.join(outputPath,processedData['sampleID'])}_tissuePointsProcessed.csv", 'r')
        print(f"{processedData['sampleID']} has already been processed! Loading data")
        processedData = loadProcessedMerfishSample(outputPath)
        return processedData
    except IOError:
        print(f"Processing {processedData['sampleID']}")
    if not os.path.exists(outputPath):
        os.makedirs(outputPath)
    ### need to update when we get better resolution information
    # resolutionRatio = visiumData['spotStartingResolution'] / templateData['startingResolution']
    processedData['derivativesPath'] = outputPath
    # processedVisium['tissueSpotBarcodeList'] = visiumData['tissueSpotBarcodeList']
    processedData['degreesOfRotation'] = int(rotation)
    sampleGauss = filters.gaussian(sampleData['imageDataGray'], sigma=2)
    tissueNormalized = cv2.normalize(sampleGauss, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    tissueHistMatch = match_histograms(tissueNormalized, templateData['rightHem'])
    tissueHistMatch = tissueHistMatch - tissueHistMatch.min()
    if 'scaleFactors' in sampleData:
        # tissue points do not necessarily start at [0,0], check bounding box
        tissuePoints = processedData['tissuePositionList']
        tissuePoints[:,0] = tissuePoints[:,0] + np.abs(sampleData['scaleFactors']['bbox_microns'][0])
        tissuePoints[:,1] = tissuePoints[:,1] + np.abs(sampleData['scaleFactors']['bbox_microns'][1])
        tissuePointsResized = tissuePoints * 0.09 # sampleData['scaleFactors']['microns_per_pixel']  
    else:
        tissuePointsResized = processedData['tissuePositionList'] * 0.09
    tissuePointsRotated, tissueRotated = rotateTissuePoints(tissuePointsResized, tissueHistMatch, rotation)
    processedData['tissueProcessed'] = tissueRotated
    processedData['geneListMasked'] = sampleData['geneList']
    processedData['processedTissuePositionList'] = tissuePointsRotated
    if log2normalize==True:
        processedData['geneMatrixLog2'] = sp_sparse.csc_matrix(np.log2((processedData['geneMatrix'] + 1)))
        sp_sparse.save_npz(f"{os.path.join(outputPath,processedData['sampleID'])}_tissuePointOrderedGeneMatrixLog2Normalized.npz", processedData['geneMatrixLog2'])
        
    else:
        sp_sparse.save_npz(f"{os.path.join(outputPath,processedData['sampleID'])}_tissuePointOrderedGeneMatrix.npz", processedData['geneMatrix'])
            
    # write outputs
    # writes json containing general info and masked gene list
    processedData['locProcessedTissuePointsCSV']  = f"{os.path.join(outputPath,processedData['sampleID'])}_tissuePointsProcessed.csv"
    processedDataDict = {
        "sampleID": sampleData['sampleID'],
        "rotation": int(rotation),
        "locProcessedTissuePointsCSV": processedData['locProcessedTissuePointsCSV'],
        "geneList": processedData['geneList']
    }
        # Serializing json
    json_object = json.dumps(processedDataDict, indent=4)
     
    # # Writing to sample.json
    with open(f"{processedData['derivativesPath']}/{processedData['sampleID']}_processing_information.json", "w") as outfile:
        outfile.write(json_object)
    # writes sorted, masked, normalized filtered feature matrix to .npz file
    
    # writes image for masked greyscale tissue, as well as the processed image that will be used in registration
    cv2.imwrite(f"{processedData['derivativesPath']}/{processedData['sampleID']}_tissue.png",255*sampleData['imageDataGray'])
    cv2.imwrite(f"{processedData['derivativesPath']}/{processedData['sampleID']}_tissueProcessed.png",processedData['tissueProcessed'])
    
    pts = {'x': processedData['processedTissuePositionList'][:,1], 'y': processedData['processedTissuePositionList'][:,0]}
    pts = pd.DataFrame(pts)
    pts.to_csv(processedData['locProcessedTissuePointsCSV'], index=False)
    # header=['x','y','z','t','label','comment']
    # rowFormat = []
    # # the x and y are swapped between ants and numpy, but this is so far dealt with within the code
    # with open(f"{os.path.join(outputPath,processedData['sampleID'])}_tissuePointsProcessed.csv", 'w', encoding='UTF8') as f:
    #     writer = csv.writer(f)
    #     writer.writerow(header)
    #     for i in range(len(processedData['processedTissuePositionList'])):
    #         rowFormat = [processedData['processedTissuePositionList'][i,1]] + [processedData['processedTissuePositionList'][i,0]] + [0] + [0] + [0] + [0]
    #         writer.writerow(rowFormat)
    return processedData

#%% registration functions
def runANTsToAllenRegistration(processedData, templateData, log2normalize=True, hemisphere='wholeBrain', displayImage=False):
    """
    Register spatial transcriptomic data to template data
    ----------
    Parameters:
    processedData : dict output from processXData
    templateData : dict output from chooseTemplateSlice
    log2normalize : bool
        Tells function whether to log2normalize gene matrix, default=True
    hemisphere : str
        String identifying which hemisphere of the atlas to align to. 'leftHem', 'rightHem', 'wholeBrain' default='wholeBrain'
    """
    registeredData = {}
    templateAntsImage = ants.from_numpy(templateData[hemisphere])
    maxWidth = templateData[hemisphere].shape[1]
    try:
        # checks for and loads registered tissue points if they exist
        csvfile = open(f"{os.path.join(processedData['derivativesPath'],processedData['sampleID'])}_tissuePointsProcessedToAllen.csv", 'r', newline='')
        transformedTissuePositionList = []
        with csvfile:
                csvreader = csv.reader(csvfile, delimiter=',')
                next(csvreader)
                for row in csvreader:
                    transformedTissuePositionList.append(row)
        tissuePointsTransformed = pd.DataFrame({'x': transformedTissuePositionList[:,0],'y': transformedTissuePositionList[:,1]})
        print(f"{processedData['sampleID']} has already been processed and is located at: {processedData['derivativesPath']}")
        print(f"Loading data for {processedData['sampleID']}")
        registeredData['sampleID'] = processedData['sampleID']
        registeredData['derivativesPath'] = processedData['derivativesPath']
        registeredData['locProcessedTissuePointsCSV'] = processedData['locProcessedTissuePointsCSV']
        registeredData['geneListMasked'] = processedData['geneListMasked']
        registeredData['fwdtransforms'] = [os.path.join(processedData['derivativesPath'],f"{processedData['sampleID']}_xfm1Warp.nii.gz"),os.path.join(processedData['derivativesPath'],f"{processedData['sampleID']}_xfm0GenericAffine.mat")]
        registeredData['invtransforms'] = [os.path.join(processedData['derivativesPath'],f"{processedData['sampleID']}_xfm0GenericAffine.mat"), os.path.join(processedData['derivativesPath'],f"{processedData['sampleID']}_xfm1InverseWarp.nii.gz"),]
        registeredData['tissueRegistered'] = plt.imread(os.path.join(processedData['derivativesPath'],f"{processedData['sampleID']}_tissue_registered_to_Allen_slice_{templateData['sliceNumber']}.png"))
    except Exception as e:
        print(f"Registering {processedData['sampleID']}")
        registeredData['sampleID'] = processedData['sampleID']
        registeredData['derivativesPath'] = processedData['derivativesPath']
        registeredData['geneListMasked'] = processedData['geneListMasked']
        
        sampleAntsImage = ants.from_numpy(processedData['tissueProcessed'])
        # run registration
        synXfm = ants.registration(fixed=templateAntsImage, moving=sampleAntsImage, \
            type_of_transform='SyNAggro', grad_step=0.1, reg_iterations=(120, 100,80,60,40,20,0), \
            syn_sampling=32, flow_sigma=3,syn_metric='mattes', outprefix=os.path.join(processedData['derivativesPath'],f"{processedData['sampleID']}_xfm"))
        
        registeredData['fwdtransforms'] = synXfm['fwdtransforms']
        registeredData['invtransforms'] = synXfm['invtransforms']
        # apply syn transform to tissue spot coordinates
        ptsToTransform = pd.DataFrame({'x': processedData['processedTissuePositionList'][:,1], 'y': processedData['processedTissuePositionList'][:,0]})
        tissuePointsTransformed = ants.apply_transforms_to_points(2, ptsToTransform,synXfm['fwdtransforms'])
        registeredData['tissueRegistered'] = synXfm["warpedmovout"].numpy()
    
    registeredData['transformedTissuePositionList'] = np.array([tissuePointsTransformed.y.to_numpy(),tissuePointsTransformed.x.to_numpy()], dtype='float32').transpose()
    # mask out spots that fall outside of the template mask
    transformedTissuePositionListMask = np.logical_and(registeredData['transformedTissuePositionList'] > 0, registeredData['transformedTissuePositionList'] < maxWidth)
    maskedTissuePositionList = []
    geneMatrixMaskedIdx = []
    for i, masked in enumerate(transformedTissuePositionListMask):
        if masked.all() == True:
            geneMatrixMaskedIdx.append(i)
            maskedTissuePositionList.append(registeredData['transformedTissuePositionList'][i])

    registeredData['maskedTissuePositionList'] = np.array(maskedTissuePositionList, dtype='float32')

    if log2normalize == True:
        tempDenseMatrix = processedData['geneMatrixLog2'].todense().astype('float32')
    else:
        tempDenseMatrix = processedData['geneMatrix'].todense().astype('float32')
    registeredData['geneMatrixMasked'] = sp_sparse.csc_matrix(tempDenseMatrix[:,geneMatrixMaskedIdx])
    if displayImage == True:
        plt.figure()
        plt.imshow(registeredData['tissueRegistered'], cmap='gray')
        plt.show()

    # write re-ordered gene matrix csv to match tissue spot order
    registeredData['locRegisteredTissuePointsCSV'] = f"{os.path.join(registeredData['derivativesPath'],registeredData['sampleID'])}_tissuePointsProcessedToAllen.csv"
    tissuePointsTransformed.to_csv(registeredData['locRegisteredTissuePointsCSV'], index=False)
    sp_sparse.save_npz(f"{os.path.join(processedData['derivativesPath'],processedData['sampleID'])}_OrderedLog2FeatureMatrixTemplateMasked.npz", sp_sparse.csc_matrix(registeredData['geneMatrixMasked']))        
    cv2.imwrite(f"{registeredData['derivativesPath']}/{registeredData['sampleID']}_tissue_registered_to_Allen_slice_{templateData['sliceNumber']}.png",registeredData['tissueRegistered'])
    
    return registeredData

def runANTsInterSampleRegistration(processedData, sampleToRegisterTo, log2normalize=True, displayImage=False):
    """ 
    Register one processed spatial transcriptomic sample to another
    ----------
    Parameters:
    processedData : dict output from processXData
    sampleToRegisterTo : dict output from processXData
    log2normalized : bool
        bool identifying whether log2normalization was performed default=True
    """
    registeredData = {}
    registeredData['sampleID'] = processedData['sampleID']
    registeredData['derivativesPath'] = processedData['derivativesPath']
    registeredData['locProcessedTissuePointsCSV'] = processedData['locProcessedTissuePointsCSV']
    registeredData['geneListMasked'] = processedData['geneListMasked']
    registeredData['locProcessedTissuePointsToSampleCSV'] = os.path.join(f"{os.path.join(processedData['derivativesPath'],processedData['sampleID'])}_to_{sampleToRegisterTo['sampleID']}TransformApplied.csv")
    transformedTissuePositionList = []
    try:
        # checks for and loads registered tissue points if they exist
        csvfile = open(registeredData['locProcessedTissuePointsToSampleCSV'], 'r', newline='')
        print(f"{processedData['sampleID']} has already been registered to {sampleToRegisterTo['sampleID']}")
        print(f"Loading data for {processedData['sampleID']}")
        with csvfile:
                csvreader = csv.reader(csvfile, delimiter=',')
                next(csvreader)
                for row in csvreader:
                    transformedTissuePositionList.append(row)
        tissuePointsTransformed = pd.DataFrame({'x': transformedTissuePositionList[:,0],'y': transformedTissuePositionList[:,1]})
        # This syntax should work correctly, will want to double check
        registeredData['fwdtransforms'] = [os.path.join(processedData['derivativesPath'],f"{processedData['sampleID']}_xfm1Warp.nii.gz"),os.path.join(processedData['derivativesPath'],f"{processedData['sampleID']}_xfm0GenericAffine.mat")]
        registeredData['invtransforms'] = [os.path.join(processedData['derivativesPath'],f"{processedData['sampleID']}_xfm0GenericAffine.mat"), os.path.join(processedData['derivativesPath'],f"{processedData['sampleID']}_xfm1InverseWarp.nii.gz"),]
        registeredData['tissueRegistered'] = plt.imread(os.path.join(f"{registeredData['derivativesPath']}/{registeredData['sampleID']}_registered_to_{sampleToRegisterTo['sampleID']}.png"))
    except Exception as e:
        templateAntsImage = ants.from_numpy(sampleToRegisterTo['tissueProcessed'])
        sampleAntsImage = ants.from_numpy(processedData['tissueProcessed'])
        # mattes seems to be most conservative syn_metric
        synXfm = ants.registration(fixed=templateAntsImage, moving=sampleAntsImage, \
            type_of_transform='SyNAggro', grad_step=0.1, reg_iterations=(120, 100,80,60,40,20,0), \
            syn_sampling=32, flow_sigma=3, syn_metric='mattes', outprefix=os.path.join(processedData['derivativesPath'],f"{processedData['sampleID']}_to_{sampleToRegisterTo['sampleID']}_xfm"))
        
        ptsToTransform = pd.DataFrame({'x': processedData['processedTissuePositionList'][:,1], 'y': processedData['processedTissuePositionList'][:,0]})
        # apply syn transform to tissue spot coordinates
        tissuePointsTransformed = ants.apply_transforms_to_points(2, ptsToTransform,synXfm['invtransforms'])
        tissuePointsTransformed.to_csv(registeredData['locProcessedTissuePointsToSampleCSV'], index=False)

        registeredData['tissueRegistered'] = synXfm["warpedmovout"].numpy()
        registeredData['fwdtransforms'] = synXfm['fwdtransforms']
        registeredData['invtransforms'] = synXfm['invtransforms']
    registeredData['transformedTissuePositionList'] = np.array([tissuePointsTransformed.y.to_numpy(),tissuePointsTransformed.x.to_numpy()], dtype='float32').transpose()
        
    if log2normalize==True:
        registeredData['geneMatrixLog2'] = processedData['geneMatrixLog2']
    else:
        registeredData['geneMatrix'] = processedData['geneMatrix']
        
    if displayImage==True:
        plt.figure()
        plt.imshow(registeredData['tissueRegistered'], cmap='gray')
        plt.scatter(registeredData['transformedTissuePositionList'][0:,0],registeredData['transformedTissuePositionList'][0:,1], marker='.', c='red', alpha=0.3)
        plt.show()

    cv2.imwrite(f"{registeredData['derivativesPath']}/{registeredData['sampleID']}_registered_to_{sampleToRegisterTo['sampleID']}.png",registeredData['tissueRegistered'])

    return registeredData
    
def applyAntsTransformations(registeredST, bestSampleRegisteredToTemplate, templateData, log2normalize=True, hemisphere='wholeBrain', displayImage=False):
    """
    Apply transformation created during template registration to tissue points
    
    Parameters
    ----------
    registeredST : dict
        Sample to apply transformation to, output of runAntsInterSampleRegistration
    bestSampleRegisteredToTemplate : dict
        Best fit sample registered to template image
    templateData : dict
        Template data
    log2normalize : bool, optional
        Whether to apply log2normalization. The default is True.
    hemisphere : bool, optional
        Which brain hemisphere to align to. The default is 'wholeBrain'.

    Returns
    -------
    templateRegisteredData : dict
        Similar structure as input data, but with registered image and spots

    """
    templateRegisteredData = {}
    transformedTissuePositionList = []
    templateRegisteredData['derivativesPath'] = registeredST['derivativesPath']
    templateRegisteredData['sampleID'] = registeredST['sampleID']
    templateRegisteredData['bestFitSampleID'] = bestSampleRegisteredToTemplate['sampleID']
    templateRegisteredData['geneListMasked'] = registeredST['geneListMasked']
    templateRegisteredData['locTissuePointsToAllen'] = f"{os.path.join(registeredST['derivativesPath'],registeredST['sampleID'])}_to_{bestSampleRegisteredToTemplate['sampleID']}TemplateTransformApplied.csv"
    imageFilename = os.path.join(registeredST['derivativesPath'],f"{registeredST['sampleID']}_registered_to_{bestSampleRegisteredToTemplate['sampleID']}_to_Allen_slice_{templateData['sliceNumber']}.png")
    try:
        # checks for and loads registered tissue points if they exist
        csvfile = open(registeredST['locTissuePointsToAllen'], 'r', newline='')
        with csvfile:
                csvreader = csv.reader(csvfile, delimiter=',')
                next(csvreader)
                for row in csvreader:
                    transformedTissuePositionList.append(row)
        transformedTissuePositionList = pd.DataFrame({'x': transformedTissuePositionList[:,0],'y': transformedTissuePositionList[:,1]})
        templateRegisteredData['tissueRegistered'] = plt.imread(imageFilename)
    except Exception as e:
        templateAntsImage = ants.from_numpy(templateData[hemisphere])
        sampleAntsImage = ants.from_numpy(registeredST['tissueRegistered'])
        sampleToTemplate = ants.apply_transforms( fixed=templateAntsImage, moving=sampleAntsImage, transformlist=bestSampleRegisteredToTemplate['fwdtransforms'])
        
        # applies the image transformation to the tissue point coordinates
        ptsToTransform = pd.DataFrame({'x': registeredST['transformedTissuePositionList'][:,1], 'y': registeredST['transformedTissuePositionList'][:,0]})
        tissuePointsTransformed = ants.apply_transforms_to_points(2, ptsToTransform,bestSampleRegisteredToTemplate['invtransforms'])
        tissuePointsTransformed.to_csv(registeredST['locProcessedTissuePointsToSampleCSV'], index=False)
        
        with open(registeredST['locProcessedTissuePointsToSampleCSV'], newline='') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',')
            next(csvreader)
            for row in csvreader:
                transformedTissuePositionList.append(row)
                
        templateRegisteredData['tissueRegistered'] = sampleToTemplate.numpy()
        cv2.imwrite(imageFilename, templateRegisteredData['tissueRegistered'])
    transformedTissuePositionList = np.array(transformedTissuePositionList, dtype='float32')
    # switching x,y columns back to python compatible and deleting empty columns
    transformedTissuePositionList[:,[0,1]] = transformedTissuePositionList[:,[1,0]]

    if displayImage==True:
        plt.figure()
        plt.imshow(templateRegisteredData['tissueRegistered'], cmap='gray')
        plt.scatter(transformedTissuePositionList[0:,0],transformedTissuePositionList[0:,1], marker='.', c='red', alpha=0.3)
        plt.show()
        
    transformedTissuePositionListMask = np.logical_and(transformedTissuePositionList > 0, transformedTissuePositionList < templateRegisteredData['tissueRegistered'].shape[0])
    maskedTissuePositionList = []
    geneMatrixMaskedIdx = []
    for i, masked in enumerate(transformedTissuePositionListMask):
        if masked.all() == True:
            geneMatrixMaskedIdx.append(i)
            maskedTissuePositionList.append(transformedTissuePositionList[i])
    templateRegisteredData['maskedTissuePositionList'] = np.array(maskedTissuePositionList, dtype='float32')
    if log2normalize == True:
        tempDenseMatrix = registeredST['geneMatrixLog2'].todense().astype('float32')
    else:
        tempDenseMatrix = registeredST['geneMatrix'].todense().astype('float32')
    templateRegisteredData['geneMatrixMasked'] = sp_sparse.csr_matrix(tempDenseMatrix[:,geneMatrixMaskedIdx])

    return templateRegisteredData

#%% digital spot functions
def createDigitalSpots(templateRegisteredData, desiredSpotSize, displayImage=False):
    """ 
    Create digital spots of a particular spacing for a spatial transcriptomic sample
    ----------
    This function is used to create honeycomb packed spots spaced at (10um * desiredSpotSize)
    on center from each other. 
    ----------
    Parameters:
    templateRegisteredData : dict output from processXData, runANTsToAllenRegistration, runANTsInterSampleRegistration, etc
    desiredSpotSize : int
        Value defining the distance between spots where final distance is 10*desiredSpotSize micron on center
    ----------
    Returns
    digitalSpots : array of [x,y] coordinates for digital spots
    """
    w = np.sqrt(3) * (desiredSpotSize/2)   # width of pointy up hexagon
    h = desiredSpotSize    # height of pointy up hexagon
    currentX = 0
    currentY = 0
    rowCount = 0
    templateSpots = []
    while currentY < templateRegisteredData['tissueRegistered'].shape[0]:
        if currentX < templateRegisteredData['tissueRegistered'].shape[1]:
            templateSpots.append([currentX, currentY])
            currentX += w
        elif (currentX > templateRegisteredData['tissueRegistered'].shape[1]):
            rowCount += 1
            currentY += h * (3/4)
            if ((currentY < templateRegisteredData['tissueRegistered'].shape[0]) and (rowCount % 2)):
                currentX = w/2
            else:
                currentX = 0
        elif ((currentX > templateRegisteredData['tissueRegistered'].shape[1] * 10) and (currentY > templateRegisteredData['tissueRegistered'].shape[0] * 10)):
            print("something is wrong")

    templateSpots = np.array(templateSpots)

    roundedTemplateSpots = np.array(templateSpots.round(), dtype='int32')
    digitalSpots = []
    for row in range(len(roundedTemplateSpots)):
        if templateRegisteredData['tissueRegistered'][roundedTemplateSpots[row,1],roundedTemplateSpots[row,0]] > 0:
            digitalSpots.append(templateSpots[row])
            
    digitalSpots = np.array(digitalSpots)
    if displayImage==True:
        plt.figure()
        plt.imshow(templateRegisteredData['tissueRegistered'])
        plt.scatter(digitalSpots[:,0],digitalSpots[:,1], alpha=0.3)
        plt.show()
    # write csv of digital spots in ants format
    derivativesPath = templateRegisteredData['derivativesPath'].split('/')
    del derivativesPath[-1]
    derivativesPath = os.path.join('/',*derivativesPath)
    digitalSpotsDF = pd.DataFrame({'x': digitalSpots[:,1], 'y': digitalSpots[:,0]})
    digitalSpotsDF.to_csv(os.path.join(derivativesPath,'digitalSpotCoordinates.csv'), index=False)
    return digitalSpots

def findDigitalNearestNeighbors(digitalSpots, templateRegisteredSpots, kNN, spotDist):
    """ 
    find nearest neighbor digital spots for each spot in a spatial transcriptomic sample
    ----------
    Parameters:
    digitalSpots : list of coordinates
        List of coordinates given as an output of createDigitalSpots
    templateRegisteredSpots : list of coordinates
        List of template registered spot coordinates
    kNN : number of desired nearest neighbors per spot
    spotDist : int
        Value defining the distance between spots where final distance is 10*desiredSpotSize micron on center
    """
    allSpotNN = []
    allMeanCdists = []
    blankIdx = np.full(kNN, -9999, dtype='int32')
    
    spotCdist = sp_spatial.distance.cdist(templateRegisteredSpots, digitalSpots, 'euclidean')
    sortedSpotCdist = np.argsort(spotCdist, axis=0)
    actSpotCdist = np.array(np.transpose(sortedSpotCdist[0:kNN,:]), dtype='int32')
    spotNNIdx = []
    for spotNum,spotIdx in enumerate(actSpotCdist):
        spotMeanCdist = np.mean(spotCdist[spotIdx,spotNum])
        if spotMeanCdist < (spotDist * 3):
            if len(allSpotNN) == 0:
                allSpotNN = np.transpose(spotIdx)
                allMeanCdists = spotMeanCdist
                # spotNNIdx = np.array(spotNNIdx, dtype='int32')
            else:
                # print(spotIdx)
                allSpotNN = np.append(allSpotNN, np.transpose(spotIdx))
                allMeanCdists = np.append(allMeanCdists, spotMeanCdist)
           
        else:
            allMeanCdists = np.append(allMeanCdists, spotMeanCdist)
            allSpotNN = np.append(allSpotNN, blankIdx, axis=0)
    
    allMeanCdists = np.array(allMeanCdists, dtype='int32')
    allSpotNN = np.array(allSpotNN,dtype='int32')
    allSpotNN = np.reshape(allSpotNN, [-1,kNN])
    return allSpotNN, allMeanCdists

def annotateDigitalSpots(digitalSpots, templateSlice, hemisphere='wholeBrain'):
    """    
    Parameters
    ----------
    digitalSpots : float array
        Array of digital spot coordinates from createDigitalSpots().
    templateSlice : Allen CCF template data
        Template data used during registration, from chooseTemplateSlice().
    hemisphere : str, optional
        String describing hemisphere of brain
        'wholeBrain','leftHem','rightHem'. The default is 'wholeBrain'.
    ----------
    Returns
    ----------
    annotatedDigitalSpots : float array
        An array containing the digital spots along with their annotation.

    """
    roundedSpots = np.rint(digitalSpots)
    templateAnnot = templateSlice['wholeBrainAnnot']
    if hemisphere=='leftHem':
        templateAnnot = templateSlice['leftHemAnnot']
    elif hemisphere=='rightHem':
        templateAnnot = templateSlice['rightHemAnnot']
    spotAnnot = []
    for i in range(roundedSpots.shape[0]):
        annotationID = templateAnnot[int(roundedSpots[i,1]),int(roundedSpots[i,0])]
        try:
            annotationNameIDX = templateSlice['annotationID'].index(str(int(annotationID)))
            annotationName = templateSlice['annotationName'][annotationNameIDX]
        except ValueError:
            annotationName = 'Out of bounds'
        spotAnnot.append([digitalSpots[i,0], digitalSpots[i,1],int(annotationID), annotationName])
    spotAnnot = np.array(spotAnnot)
    return spotAnnot
#%% load functions
def loadProcessedVisiumSample(locOfProcessedSample, loadLog2Norm=True):
    """   
    Load previously processed visium sample
    ----------
    Parameters
    ----------
    locOfProcessedSample : directory path
        Path location of processed sample
    loadLog2Norm : bool
        Whether or not to load the log2 or non-normalized gene matrix
    ----------
    Returns
    -------
    processedData : dict
        Dictionary containing processed data

    """
    processedVisium = {}
    processedVisium['derivativesPath'] = os.path.join(locOfProcessedSample)
    processedVisium['sampleID'] = locOfProcessedSample.rsplit(sep='/',maxsplit=1)[-1]
    processedVisium['sourceType'] = 'visium'
    processedVisium['tissueProcessed'] = io.imread(os.path.join(processedVisium['derivativesPath'],f"{processedVisium['sampleID']}_tissueProcessed.png"))
    jsonPath = open(os.path.join(processedVisium['derivativesPath'],f"{processedVisium['sampleID']}_processing_information.json"))
    processedSampleJson = json.loads(jsonPath.read())
    processedVisium['geneListMasked'] = processedSampleJson['geneList']
    processedVisium['spotCount'] = processedSampleJson['spotCount']
    processedVisium['resolutionRatio'] = processedSampleJson['resolutionRatio']
    processedVisium['locProcessedTissuePointsCSV'] = os.path.join(f"{os.path.join(processedVisium['derivativesPath'],processedVisium['sampleID'])}_tissuePointsProcessed.csv")
    if loadLog2Norm==True:
        geneMatrixLog2 = sp_sparse.load_npz(os.path.join(processedVisium['derivativesPath'], f"{processedVisium['sampleID']}_tissuePointOrderedFeatureMatrixLog2Normalized.npz"))
        processedVisium['geneMatrixLog2'] = geneMatrixLog2
    else:
        geneMatrix = sp_sparse.load_npz(os.path.join(processedVisium['derivativesPath'], f"{processedVisium['sampleID']}_tissuePointOrderedFeatureMatrix.npz"))
        processedVisium['geneMatrix'] = geneMatrix
    tissuePositionList = []
    csvfile = open(processedVisium['locProcessedTissuePointsCSV'], newline='') 
    with csvfile:
            csvreader = csv.reader(csvfile, delimiter=',')
            next(csvreader)
            for row in csvreader:
                tissuePositionList.append(row)
                
    tissuePositionList = np.array(tissuePositionList, dtype='float32')
    # switching x,y columns back to python compatible and deleting empty columns
    tissuePositionList[:,[0,1]] = tissuePositionList[:,[1,0]]
    processedVisium['processedTissuePositionList'] = tissuePositionList
    return processedVisium

def loadProcessedMerfishSample(locOfProcessedSample, loadLog2Norm=True):
    """   
    Load previously processed merfish sample
    ----------
    Parameters
    ----------
    locOfProcessedSample : directory path
        Path location of processed sample
    loadLog2Norm : bool
        Whether or not to load the log2 or non-normalized gene matrix
    ----------
    Returns
    -------
    processedData : dict
        Dictionary containing processed data

    """
    processedSample = {}
    processedSample['derivativesPath'] = os.path.join(locOfProcessedSample)
    processedSample['sampleID'] = locOfProcessedSample.rsplit(sep='/',maxsplit=1)[-1]
    processedSample['sourceType'] = 'merfish'
    processedSample['tissueProcessed'] = io.imread(os.path.join(processedSample['derivativesPath'],f"{processedSample['sampleID']}_tissueProcessed.png"))
    jsonPath = open(os.path.join(processedSample['derivativesPath'],f"{processedSample['sampleID']}_processing_information.json"))
    processedSampleJson = json.loads(jsonPath.read())
    processedSample['geneListMasked'] = processedSampleJson['geneList']
    processedSample['degreesOfRotation'] = processedSampleJson['rotation']
    # need to consider merfish alternative to spotCount, like cell count
    # processedSample['spotCount'] = processedSampleJson['spotCount']
    if loadLog2Norm==True:
        geneMatrixLog2 = sp_sparse.load_npz(os.path.join(processedSample['derivativesPath'], f"{processedSample['sampleID']}_tissuePointOrderedGeneMatrixLog2Normalized.npz"))
        processedSample['geneMatrixLog2'] = geneMatrixLog2
    else:
        geneMatrix = sp_sparse.load_npz(os.path.join(processedSample['derivativesPath'], f"{processedSample['sampleID']}_tissuePointOrderedGeneMatrix.npz"))
        processedSample['geneMatrix'] = geneMatrix
    tissuePositionList = []
    with open(os.path.join(f"{os.path.join(processedSample['derivativesPath'],processedSample['sampleID'])}_tissuePointsProcessed.csv"), newline='') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',')
            next(csvreader)
            for row in csvreader:
                tissuePositionList.append(row)
                
    tissuePositionList = np.array(tissuePositionList, dtype='float32')
    # switching x,y columns back to python compatible
    tissuePositionList[:,[0,1]] = tissuePositionList[:,[1,0]]
    return processedSample

def loadAllenRegisteredSample(locOfRegSample, log2normalize=True):
    templateRegisteredData = {}
    templateRegisteredData['derivativesPath'] = os.path.join(locOfRegSample)
    templateRegisteredData['sampleID'] = locOfRegSample.rsplit(sep='/',maxsplit=1)[-1]
    try:
        bestFitSample = glob(os.path.join(locOfRegSample, f"{templateRegisteredData['sampleID']}_registered_to_*_to_Allen_slice_*.png"))
        bestFitSample = bestFitSample[0]
    except IndexError:
        print(f"No registered data found in {locOfRegSample}")
    templateRegisteredData['tissueRegistered'] = io.imread(bestFitSample)
    bestFitSample = bestFitSample.rsplit(sep='/',maxsplit=1)[-1]
    # id of best fit is the third from the end
    bestFitSample = bestFitSample.rsplit(sep='_')[-3]
    templateRegisteredData['bestFitSampleID'] = bestFitSample
    jsonPath = open(os.path.join(templateRegisteredData['derivativesPath'],f"{templateRegisteredData['sampleID']}_processing_information.json"))
    processedSampleJson = json.loads(jsonPath.read())
    templateRegisteredData['geneListMasked'] = processedSampleJson['geneList']
    templateRegisteredData['spotCount'] = processedSampleJson['spotCount']
    transformedTissuePositionList = []
    with open(os.path.join(f"{os.path.join(templateRegisteredData['derivativesPath'],templateRegisteredData['sampleID'])}_to_{templateRegisteredData['bestFitSampleID']}TemplateTransformApplied.csv"), newline='') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',')
            next(csvreader)
            for row in csvreader:
                transformedTissuePositionList.append(row)
    transformedTissuePositionList = np.array(transformedTissuePositionList, dtype='float32')
    transformedTissuePositionList[:,[0,1]] = transformedTissuePositionList[:,[1,0]]
    transformedTissuePositionListMask = []
    transformedTissuePositionListMask = np.logical_and(transformedTissuePositionList > 0, transformedTissuePositionList < templateRegisteredData['tissueRegistered'].shape[0])
    maskedTissuePositionList = []
    geneMatrixMaskedIdx = []
    for i, masked in enumerate(transformedTissuePositionListMask):
        if masked.all() == True:
            geneMatrixMaskedIdx.append(i)
            maskedTissuePositionList.append(transformedTissuePositionList[i])
            # filteredFeatureMatrixMasked = np.append(filteredFeatureMatrixMasked, registeredVisium['filteredFeatureMatrixOrdered'][:,i],axis=1)
    templateRegisteredData['maskedTissuePositionList'] = np.array(maskedTissuePositionList, dtype='float32')
    if log2normalize == True:
        geneMatrixLog2 = sp_sparse.load_npz(os.path.join(templateRegisteredData['derivativesPath'], f"{templateRegisteredData['sampleID']}_tissuePointOrderedFeatureMatrixLog2Normalized.npz"))
        tempDenseMatrix = geneMatrixLog2.todense().astype('float32')
    else:
        geneMatrix = sp_sparse.load_npz(os.path.join(templateRegisteredData['derivativesPath'], f"{templateRegisteredData['sampleID']}_tissuePointOrderedFeatureMatrix.npz"))
        tempDenseMatrix = geneMatrix.todense().astype('float32')
    templateRegisteredData['geneMatrixMasked'] = sp_sparse.csr_matrix(tempDenseMatrix[:,geneMatrixMaskedIdx])
    return templateRegisteredData

def loadGeneListFromTxt(locOfTextFile):
    geneListFromTxt = []
    with open(locOfTextFile) as f:
        for gene in f:
            geneListFromTxt.append(gene.strip('\n'))
    return geneListFromTxt

def loadGeneListFromCsv(locOfCsvFile):
    geneListFromCsv = []
    with open(locOfCsvFile, 'r', encoding='UTF8') as f:
        sigGeneReader = csv.reader(f, delimiter=',')
        for row in sigGeneReader:
            geneListFromCsv.append(row[0])
    return geneListFromCsv

def loadParticipantsTsv(locOfTsvFile, imageList='all'):
    # as the input takes the location of participants.tsv
    # default loads all images in participants.tsv
    # using a list of int index in second position will mask to use only those samples listed
    # might want to rethink to excluding samples prior to creating participants.tsv
    # in the future, could add default to rawdata/participants.tsv
    sampleIDs = []
    sampleInfo = []
    with open(locOfTsvFile, newline='') as tsvfile:
        tsvreader = csv.reader(tsvfile, delimiter='\t')
        # assumes tsv has header of sample id, degrees of rotation, and group status
        next(tsvreader)
        for row in tsvreader:
            sampleIDs.append(row[0])
            sampleInfo.append(row[1:])
    
    sampleInfo = np.array(sampleInfo, dtype='int')

    # list of good images
    if imageList == 'all':
        imageList = list(range(len(sampleIDs)))
    # else:
    #     imageList = [0,1,2,3,4,5,6,7,10,11,12,13,15]
    
    experiment = {'sample-id': np.asarray(sampleIDs)[imageList],
                        'rotation': sampleInfo[imageList,0],
                        'experimental-group': sampleInfo[imageList,1]}
    return experiment

def setExperimentalFolder(locOfExpFolder):
    global rawdata
    global derivatives
    rawdata = os.path.join(locOfExpFolder, 'rawdata')
    derivatives = os.path.join(locOfExpFolder, 'derivatives')
    return rawdata, derivatives
    
# desired region will need to be in the naming format of the allen ccf
# should expand on information stored in the regionMask output of createRegionalMask
# include region name, parent structure, color, etc
def createRegionalMask(template, desiredRegion, hemisphere='wholeBrain', displayImage=False):
    regionIdx = template['annotationName'].index(desiredRegion)
    regionID = int(template['annotationID'][regionIdx])
    regionColor = np.append(template['annotationColor'][regionIdx],1)
    structIDs = []
    templateAnnot = template['wholeBrainAnnot']
    if hemisphere=='leftHem':
        templateAnnot = template['leftHemAnnot']
    elif hemisphere=='rightHem':
        templateAnnot = template['rightHemAnnot']
    for idx, actStruct in enumerate(np.array(template['structureIDPath'])):
        if f"/{regionID}/" in actStruct:
            structIDs.append(template['annotationID'][idx])
    structIDs = np.array(structIDs)
    regionMask = np.zeros(templateAnnot.shape, dtype='int')
    
    for actID in structIDs:
        regionBoolMask = np.where(templateAnnot == int(actID))
        if any(regionBoolMask[0]):
            regionMask[regionBoolMask[0],regionBoolMask[1]] = 1
    if displayImage==True:
        plt.figure()
        plt.imshow(template[hemisphere], cmap='gray_r')
        regionColormap = mcolors.ListedColormap([[0,0,0,0],
          regionColor])
        plt.imshow(regionMask, cmap=regionColormap, interpolation='none', alpha=0.8)
        plt.show()
    return regionMask

def createRegionalDigitalSpots(regionMask, template, desiredSpotSize, hemisphere='wholeBrain', displayImage=False):
    w = np.sqrt(3) * (desiredSpotSize/2)   # width of pointy up hexagon
    h = desiredSpotSize    # height of pointy up hexagon
    currentX = 0
    currentY = 0
    rowCount = 0
    templateImage = template['wholeBrain']
    if hemisphere=='leftHem':
        templateImage = template['leftHem']
    elif hemisphere=='rightHem':
        templateImage = template['rightHem']
    templateSpots = []
    while currentY < regionMask.shape[0]:
        if currentX < regionMask.shape[1]:
            templateSpots.append([currentX, currentY])
            currentX += w
        elif (currentX > regionMask.shape[1]):
            rowCount += 1
            currentY += h * (3/4)
            if ((currentY < regionMask.shape[0]) and (rowCount % 2)):
                currentX = w/2
            else:
                currentX = 0
        elif ((currentX > regionMask.shape[1] * 10) and (currentY > regionMask.shape[0] * 10)):
            print("something is wrong")

    templateSpots = np.array(templateSpots)
    # remove non-tissue spots
    roundedTemplateSpots = np.array(templateSpots.round(), dtype='int32')
    digitalSpots = []
    for row in range(len(roundedTemplateSpots)):
        if regionMask[roundedTemplateSpots[row,1],roundedTemplateSpots[row,0]] > 0:
            digitalSpots.append(templateSpots[row])
    digitalSpots = np.array(digitalSpots)
    
    if displayImage==True:
        plt.imshow(templateImage, cmap='gray_r')
        # plt.imshow(regionMask)
        plt.scatter(digitalSpots[:,0],digitalSpots[:,1], alpha=0.3, s=5)
        plt.show()
    return digitalSpots

def viewGeneInProcessedVisium(processedSample, geneName):
    try:
        geneIndex = processedSample['geneListMasked'].index(geneName)
        actSpots = processedSample['geneMatrixLog2'][geneIndex, :]
        plt.figure()
        plt.imshow(processedSample['tissueProcessed'], cmap='gray')
        plt.scatter(processedSample['processedTissuePositionList'][:,0],processedSample['processedTissuePositionList'][:,1], c=np.array(actSpots.todense()), alpha=0.8, cmap='Reds', s=5)
        plt.title(f'Gene count for {geneName} in {processedSample["sampleID"]}')
        plt.colorbar()
        plt.show()
    except(ValueError):
        print(f'{geneName} not found in dataset')
        
def viewGeneInRegisteredVisium(registeredSample, geneName):
    try:
        geneIndex = registeredSample['geneListMasked'].index(geneName)
        actSpots = registeredSample['geneMatrixLog2'][geneIndex, :]
        plt.figure()
        plt.imshow(registeredSample['tissueProcessed'], cmap='gray')
        plt.scatter(registeredSample['processedTissuePositionList'][:,0],registeredSample['processedTissuePositionList'][:,1], c=np.array(actSpots.todense()), alpha=0.8, cmap='Reds', marker='.')
        plt.title(f'Gene count for {geneName} in {registeredSample["sampleID"]}')
        plt.colorbar()
        plt.show()
    except(ValueError):
        print(f'{geneName} not found in dataset')

#%% add analysis that utilizes spots of known gene
# this version uses processed but unregistered samples
def selectSpotsWithGene(processedSample, geneToSelect):
    denseMatrix = processedSample['geneMatrixLog2']
    denseMatrix = denseMatrix.todense().astype('float32')
    geneIndex = processedSample['geneListMasked'].index(geneToSelect)
    actSpots = processedSample['geneMatrixLog2'][geneIndex, :]
    actSpots = actSpots.todense().astype('float32')
    posSpots = actSpots > 0
    if np.sum(actSpots) > 0:
        posSpots = np.squeeze(np.array(posSpots))
        maskedTissuePositionList = processedSample['processedTissuePositionList'][posSpots,:]
        maskedMatrix = denseMatrix[:,posSpots]
    else:
        print(f"No spots in {processedSample['sampleID']} are positive for {geneToSelect}")
    return maskedMatrix, maskedTissuePositionList

def selectGeneMatrixWithList(processedSample, listOfGenesToSelect):
    denseMatrix = processedSample['geneMatrixLog2']
    denseMatrix = denseMatrix.todense().astype('float32')
    idxList = []
    for actGene in listOfGenesToSelect:        
        geneIndex = processedSample['geneListMasked'].index(actGene)
        idxList.append(geneIndex)
    idxList = np.array(idxList, dtype='int32')
    maskedMatrix = sp_sparse.csc_matrix(denseMatrix[idxList, :])
    return maskedMatrix

def selectSpotsWithRegionMask(registeredSample, regionMask, template, hemisphere='wholeBrain', displayImage=True):
    templateImage = template['wholeBrain']
    if hemisphere=='leftHem':
        templateImage = template['leftHem']
    elif hemisphere=='rightHem':
        templateImage = template['rightHem']
    # denseMatrix = processedSample['geneMatrixLog2']
    # denseMatrix = denseMatrix.todense().astype('float32')
    idxRounded = np.array(np.round(registeredSample['maskedTissuePositionList']),dtype='int32')
    regionMaskedTissuePositionList = []
    regionMaskedIdx = [];
    for i, actIdx in enumerate(idxRounded):
        if regionMask[actIdx[1],actIdx[0]] == 1:
            regionMaskedTissuePositionList.append(registeredSample['maskedTissuePositionList'][i,:])
            regionMaskedIdx.append(i)
    regionMaskedTissuePositionList = np.array(regionMaskedTissuePositionList, dtype='float32')
    regionMaskedGeneMatrix = registeredSample['geneMatrixMaskedSorted'][:,regionMaskedIdx]
    if displayImage==True:
        plt.figure()
        plt.imshow(registeredSample['tissueRegistered'],cmap='gray')
        plt.scatter(regionMaskedTissuePositionList[:,0], regionMaskedTissuePositionList[:,1])
        plt.show()
    return regionMaskedTissuePositionList, regionMaskedGeneMatrix
#%% calculate the cosine similarity of a given matrix at the coordinates given
def cosineSimOfConnection(inputMatrix,i,j):
    I = np.ravel(inputMatrix[:,i])
    J = np.ravel(inputMatrix[:,j])
    # cs = np.sum(np.dot(I,J.transpose())) / (np.sqrt(np.sum(np.square(I)))*np.sqrt(np.sum(np.square(J))))
    cs = sp_spatial.distance.cosine(I,J)
    return cs
# measureTranscriptomicSimilar will replace cosineSimOfConnection
def measureTranscriptomicSimilarity(geneMatrix, measurement='cosine'):
    """
    measure the transcriptomic similarity/distance between two spots
    ----------
    Parameters
    ----------
    geneMatrix: float array
        2D matrix of genetic or transcriptomic data, organized [gene,spot]
    edgeList: Nx2 int list
        List of edges to be measured for distance, [[spot1,spot2],[spot1,spot3],...]
    measurement: str
        Choice of measurement metric, default='cosine'
        'cosine' - cosine similarity
        'pearson' - Pearson's R correlation
    """
    
    dataSimMatrix = []
    fullyConnectedEdges = []
    for i in range(geneMatrix.shape[1]):
        for j in range(geneMatrix.shape[1]):
            fullyConnectedEdges.append([i,j])
            
    fullyConnectedEdges = np.array(fullyConnectedEdges,dtype='int32')
    fullyConnectedEdges = np.unique(np.sort(fullyConnectedEdges, axis=1),axis=0)
    if measurement=='cosine':
        dataSimMatrix = np.zeros([geneMatrix.shape[1],geneMatrix.shape[1]])
        for i,j in fullyConnectedEdges:
            I = np.ravel(geneMatrix[:,i])
            J = np.ravel(geneMatrix[:,j])
            cs = sp_spatial.distance.cosine(I,J)
            dataSimMatrix[i,j] = float(cs)
            dataSimMatrix[j,i] = float(cs)
    # elif measurement=='pearson':
    #     for i,j in fullyConnectedEdges:
    #         I = np.ravel(geneMatrix[:,i])
    #         J = np.ravel(geneMatrix[:,j])
    #         cs = scipy.stats.pearsonr(I,J)
    #         print(i, j)
    #         dataSimMatrix[i,j] = cs[0]
    #         dataSimMatrix[j,i] = cs[0]
        # dataSimMatrix = scipy.stats.pearsonr(geneMatrix)
        # dataSimMatrix.append(cs)
    return dataSimMatrix
#%% functions for merscope data
def downsampleMerfishTiff(merfishImageFilename, outputName, scale=0.01):
    # the default scale is based on merscope claiming nanometer resolution, scaled for the ccf 10um resolution    
    img = itk.imread(merfishImageFilename)
    imgSize = itk.size(img)
    imgSpacing = itk.spacing(img)
    imgOrigin = itk.origin(img)
    imgDimension = img.GetImageDimension()
    outSize = [int(imgSize[d] * scale) for d in range(imgDimension)]
    outSpacing = [imgSpacing[d] / scale for d in range(imgDimension)]
    outOrigin = [imgOrigin[d] + 0.5 * (outSpacing[d] - imgSpacing[d])
                      for d in range(imgDimension)]
    
    interpolator = itk.LinearInterpolateImageFunction.New(img)
    
    resampled = itk.resample_image_filter(
        img,
        interpolator=interpolator,
        size=outSize,
        output_spacing=outSpacing,
        output_origin=outOrigin,
    )
    outFilename = os.path.join(outputName)
    itk.imwrite(resampled, outFilename)
    
#%% cell type identification

def selectSpotsWithGeneList(processedSample, geneList, threshold=1):
    # threshold decides percentage of genes in list necessary
    geneListIdx = []
    for actGene in geneList:
        try:
            if 'allSampleGeneList' in processedSample:
                geneListIdx.append(list(processedSample['allSampleGeneList']).index(actGene))
                actSpots = processedSample['geneMatrixMaskedSorted'][geneListIdx, :]
                actSpots = actSpots.todense().astype('float32')
                posSpots = np.sum((actSpots > 0), axis=0)
                nGenes = len(geneListIdx)
                threshVal = round(nGenes * threshold)
                posSpots = posSpots > threshVal
                # posSpots = np.count_nonzero(actSpots, axis=0)
                denseMatrix = processedSample['geneMatrixMaskedSorted'].todense().astype('float32')
                if np.sum(posSpots) > 0:
                    posSpots = np.squeeze(np.array(posSpots))
                    maskedTissuePositionList = processedSample['maskedTissuePositionList'][posSpots,:]
                    maskedMatrix = denseMatrix[:,posSpots]
                else:
                    maskedMatrix = []
                    maskedTissuePositionList = []
            else:
                geneListIdx.append(processedSample['geneListMasked'].index(actGene))
                actSpots = processedSample['geneMatrixLog2'][geneListIdx, :]
                actSpots = actSpots.todense().astype('float32')
                posSpots = np.sum((actSpots > 0), axis=0)
                nGenes = len(geneListIdx)
                threshVal = round(nGenes * threshold)
                posSpots = posSpots > threshVal
                # posSpots = np.count_nonzero(actSpots, axis=0)
                denseMatrix = processedSample['geneMatrixLog2'].todense().astype('float32')
                if np.sum(posSpots) > 0:
                    posSpots = np.squeeze(np.array(posSpots))
                    maskedTissuePositionList = processedSample['processedTissuePositionList'][posSpots,:]
                    maskedMatrix = denseMatrix[:,posSpots]
                else:
                    maskedMatrix = []
                    maskedTissuePositionList = []
        except ValueError:
            print(f"No spots are positive for {actGene}!")
   
    
   
    return maskedMatrix, maskedTissuePositionList

#%% Masking functions
class SelectUsingLasso:
    """
    Parameters
    ----------
    processedSample : takes a processed sample (visium or merfish) and plots
        to allow using the lasso tool to select spots of interest
    """

    def __init__(self, processedSample, maskName, alpha_unselected=0.1):
        self.img = processedSample['tissueProcessed']
        self.id = f"{processedSample['sampleID']}_{maskName}"
        self.derivatives = f"{processedSample['derivativesPath']}_{maskName}"
        self.fig, self.ax = plt.subplots()
        self.ax.imshow(self.img, cmap='gray')
        self.pts = self.ax.scatter(processedSample['processedTissuePositionList'][:,0], processedSample['processedTissuePositionList'][:,1])
        self.canvas = self.ax.figure.canvas
        self.alpha_unselected = alpha_unselected
        self.xys = self.pts.get_offsets()
        self.Npts = len(self.xys)

        # Ensure that we have separate colors for each object
        self.fc = self.pts.get_facecolors()
        if len(self.fc) == 0:
            raise ValueError('Collection must have a facecolor')
        elif len(self.fc) == 1:
            self.fc = np.tile(self.fc, (self.Npts, 1))

        self.lasso = LassoSelector(self.ax, onselect=self.onselect)
        self.ind = []
        self.fig.canvas.mpl_connect("key_press_event", self.accept)
        self.ax.set_title("Press enter to accept selected points.")

    def onselect(self, verts):
        path = Path(verts)
        self.ind = np.nonzero(path.contains_points(self.xys))[0]
        self.fc[:, -1] = self.alpha_unselected
        self.fc[self.ind, -1] = 1
        self.pts.set_facecolors(self.fc)
        imageX,imageY = np.meshgrid(np.arange(self.img.shape[1]),np.arange(self.img.shape[0]))
        imageX,imageY = imageX.flatten(), imageY.flatten()
        points = np.vstack((imageX, imageY)).T
        self.maskPoints = path.contains_points(points)
        self.canvas.draw_idle()

    def disconnect(self):
        self.lasso.disconnect_events()
        self.fc[:, -1] = 1
        self.pts.set_facecolors(self.fc)
        self.canvas.draw_idle()
        
    def outputMaskedSpots(self):
        self.maskedSpots = self.xys[self.ind]
        # return self.maskedSpots
    
    def outputMaskedImage(self, processedSample):
        self.imageMask = self.maskPoints.reshape((self.img.shape[0],self.img.shape[1]))
        self.maskedImage = self.img * self.imageMask
        
    def flip(self):
        self.maskedImage = self.maskedImage[:,::-1]
        self.maskedSpots = self.maskedSpots * [-1,1]
        self.maskedSpots[:,0] = self.maskedSpots[:,0] + self.maskedImage.shape[0]

    def outputMaskedSample(self, processedSample):
        processedSampleMasked = {}
        processedSampleMasked['sampleID'] = self.id
        processedSampleMasked['degreesOfRotation']  = processedSample['degreesOfRotation']
        processedSampleMasked['derivativesPath'] = self.derivatives
        processedSampleMasked['sourceType'] = processedSample['sourceType']
        if os.path.exists(processedSampleMasked['derivativesPath']):
            print(f"Sample has already been created using the name {processedSampleMasked['sampleID']}")
            print(f"Now loading {processedSampleMasked['sampleID']}")
            print("If you want to create a new sample, either rename the mask or delete previous version")
            if processedSampleMasked['sourceType'] == 'merfish':
                processedSampleMasked = loadProcessedMerfishSample(processedSampleMasked['derivativesPath'])
            elif processedSampleMasked['sourceType'] == 'visium':
                processedSampleMasked = loadProcessedVisiumSample(processedSampleMasked['derivativesPath'])
        else:
            os.makedirs(processedSampleMasked['derivativesPath'])
            processedSampleMasked['geneList'] = processedSample['geneList']
            processedSampleMasked['geneListMasked'] = processedSample['geneListMasked']
            processedSampleMasked['geneMatrix'] = processedSample['geneMatrix'][:,self.ind]
            denseMatrix = processedSample['geneMatrixLog2'].todense()
            processedSampleMasked['geneMatrixLog2'] = sp_sparse.csc_matrix(denseMatrix[:,self.ind])
            sp_sparse.save_npz(f"{os.path.join(processedSampleMasked['derivativesPath'],processedSampleMasked['sampleID'])}_tissuePointOrderedGeneMatrixLog2Normalized.npz", processedSampleMasked['geneMatrixLog2'])
            processedSampleMasked['processedTissuePositionList'] = self.maskedSpots
            processedSampleMasked['tissuePositionList'] = processedSample['tissuePositionList']
            processedSampleMasked['tissueProcessed'] = self.maskedImage
            # writes json containing general info and masked gene list
            processedDataDict = {
                "sampleID": processedSampleMasked['sampleID'],
                "rotation": int(processedSample['degreesOfRotation']),
                "geneList": processedSampleMasked['geneList']
            }
                # Serializing json
            json_object = json.dumps(processedDataDict, indent=4)
             
            # # Writing to sample.json
            with open(f"{processedSampleMasked['derivativesPath']}/{processedSampleMasked['sampleID']}_processing_information.json", "w") as outfile:
                outfile.write(json_object)
            cv2.imwrite(f"{processedSampleMasked['derivativesPath']}/{processedSampleMasked['sampleID']}_tissueProcessed.png",processedSampleMasked['tissueProcessed'])
            header=['x','y','z','t','label','comment']
            rowFormat = []
            # the x and y are swapped between ants and numpy, but this is so far dealt with within the code
            with open(f"{os.path.join(processedSampleMasked['derivativesPath'],processedSampleMasked['sampleID'])}_tissuePointsProcessed.csv", 'w', encoding='UTF8') as f:
                writer = csv.writer(f)
                writer.writerow(header)
                for i in range(len(processedSampleMasked['processedTissuePositionList'])):
                    rowFormat = [processedSampleMasked['processedTissuePositionList'][i,1]] + [processedSampleMasked['processedTissuePositionList'][i,0]] + [0] + [0] + [0] + [0]
                    writer.writerow(rowFormat)
        return processedSampleMasked
    def accept(self, event):
        global maskedSpots
        if event.key == "enter":
            # print("Selected points:")
            # print(self.xys[self.ind])
            self.disconnect()
            self.ax.set_title("")
            plt.close()
        
