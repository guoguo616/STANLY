#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""".
Created on Sep 7 2022

@author: zjpeters.
"""
# data being read in includes: json, h5, csv, nrrd, jpg, and svg
import os
from skimage import io, filters, color
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
import time
# from scipy.spatial.distance import pdist, squareform, cosine, cdist
# setting up paths
derivatives = "/home/zjpeters/Documents/visiumalignment/derivatives"
rawdata = "/home/zjpeters/Documents/visiumalignment/rawdata"
# next few lines first grabs location of main script and uses that to get the location of the reference data, i.e. one back from teh code folder
codePath = os.path.realpath(os.path.dirname(__file__))
refDataPath = codePath.split('/')
del refDataPath[-1]
refDataPath = os.path.join('/',*refDataPath)
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
    # visiumData['imageDataGray'] = 1 - visiumData['imageData'][:,:,2]
    visiumData['imageDataGray'] = 1 - color.rgb2gray(visiumData['imageData'])
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
    ara_data = ants.image_read(os.path.join(refDataPath,'data','ccf','ara_nissl_10.nrrd'))
    annotation_data = ants.image_read(os.path.join(refDataPath,'data','ccf','annotation_10.nrrd'))
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
    # changed to > for inv green channel, originally < below
    processedVisium['visiumOtsu'] = processedVisium['visiumGauss'] > processedVisium['otsuThreshold']
    processedVisium['tissue'] = np.zeros(processedVisium['visiumGauss'].shape, dtype='float32')
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
    # processedVisium['tissuePointsResizedForTransform'] = processedVisium['tissuePointsRotated'] * processedVisium['resolutionRatio']
    # processedVisium['tissuePointsResizedForTransform'][:,[0,1]] = processedVisium['tissuePointsResizedForTransform'][:,[1,0]]
    
    filteredFeatureMatrixGeneList = []
    for geneName in visiumData['filteredFeatureMatrix'][0]['name']:
        filteredFeatureMatrixGeneList.append(geneName.decode())
    processedVisium['filteredFeatureMatrixGeneList'] = filteredFeatureMatrixGeneList

    # this orders the filtered feature matrix so that the columns are in the order of the coordinate list, so barcodes no longer necessary
    filteredFeatureMatrixBarcodeList = []
    for barcodeID in visiumData['filteredFeatureMatrix'][1]:
        filteredFeatureMatrixBarcodeList.append(barcodeID.decode())
    processedVisium['filteredFeatureMatrixBarcodeList'] = filteredFeatureMatrixBarcodeList 

    tissueSpotBarcodeListSorted = []
    for actbarcode in processedVisium['tissueSpotBarcodeList']:
        tissueSpotBarcodeListSorted.append(processedVisium['filteredFeatureMatrixBarcodeList'].index(actbarcode))
    # no need to keep the dense version in the processedVisium dictionary since the necessary information is in the ordered matrix and coordinates
    processedVisium['tissueSpotBarcodeListSorted'] = tissueSpotBarcodeListSorted
    denseMatrix = visiumData["filteredFeatureMatrix"][2]
    denseMatrix = denseMatrix.todense()
    orderedDenseMatrix = denseMatrix[:,tissueSpotBarcodeListSorted]
    countsPerSpot = np.sum(orderedDenseMatrix,axis=0)
    countsPerSpotMean = np.mean(countsPerSpot)
    countsPerSpotStD = np.std(countsPerSpot)
    countsPerSpotZscore = (countsPerSpot - countsPerSpotMean) / countsPerSpotStD
    spotMask = countsPerSpot > 5000
    spotMask = np.squeeze(np.array(spotMask))
    countsPerGene = np.count_nonzero(np.array(orderedDenseMatrix),axis=1, keepdims=True)
    geneMask = countsPerGene > 30
    geneMask = np.squeeze(np.array(geneMask))    
    wholeGeneList = np.array(processedVisium['filteredFeatureMatrixGeneList'])
    processedVisium['geneListMasked'] = wholeGeneList[geneMask].tolist()
    orderedDenseMatrixSpotMasked = orderedDenseMatrix[:,spotMask]
    orderedDenseMatrixSpotMasked = orderedDenseMatrix[geneMask,:]
    processedVisium['spotCount'] = orderedDenseMatrixSpotMasked.shape[1]
    print(f"{processedVisium['sampleID']} has {processedVisium['spotCount']} spots")
    processedVisium['countMaskedTissuePositionList'] = processedVisium['tissuePointsResized'][spotMask,:]
    processedVisium['filteredFeatureMatrixOrdered'] = sp_sparse.csc_matrix(orderedDenseMatrixSpotMasked)
    processedVisium['filteredFeatureMatrixLog2'] = sp_sparse.csc_matrix(np.log2((orderedDenseMatrixSpotMasked + 1)))
    
    finiteMin=np.min(countsPerSpotZscore)
    finiteMax=np.max(countsPerSpotZscore)
    zeroCenteredCmap = mcolors.TwoSlopeNorm(0,vmin=finiteMin, vmax=finiteMax)
    # plt.imshow( processedVisium['tissueRotated'], cmap='gray')
    # plt.scatter(processedVisium['tissuePointsResized'][:,0],processedVisium['tissuePointsResized'][:,1], c=np.array(countsPerSpot), alpha=0.8)
    # plt.title(f"Total gene count per spot for {processedVisium['sampleID']}")
    # plt.colorbar()
    # plt.show()
    plt.imshow( processedVisium['tissueRotated'], cmap='gray')
    plt.scatter(processedVisium['tissuePointsResized'][:,0],processedVisium['tissuePointsResized'][:,1], c=np.array(countsPerSpotZscore), alpha=0.8, cmap='seismic', norm=zeroCenteredCmap, marker='.')
    plt.title(f"Z-score of overall gene count per spot for {processedVisium['sampleID']}")
    plt.colorbar()
    plt.show()
    
    # write outputs
    sp_sparse.save_npz(f"{os.path.join(outputPath,processedVisium['sampleID'])}_tissuePointOrderedFeatureMatrix.npz", processedVisium['filteredFeatureMatrixOrdered'])
            
    cv2.imwrite(f"{processedVisium['derivativesPath']}/{processedVisium['sampleID']}_tissue.png",processedVisium['tissue'])
    
    header=['x','y','z','t','label','comment']
    rowFormat = []
    with open(f"{os.path.join(outputPath,processedVisium['sampleID'])}_tissuePointsResizeToTemplate.csv", 'w', encoding='UTF8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i in range(len(processedVisium['countMaskedTissuePositionList'])):
            rowFormat = [processedVisium['countMaskedTissuePositionList'][i,1]] + [processedVisium['countMaskedTissuePositionList'][i,0]] + [0] + [0] + [0] + [0]
            writer.writerow(rowFormat)
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
    syn_sampling=32, flow_sigma=3,syn_metric='mattes', outprefix=os.path.join(processedVisium['derivativesPath'],f"{processedVisium['sampleID']}_xfm"))
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
                #####################
                # should be able to fix with:
                # transformedTissuePositionList.append(row[1,0])
                # then remove the two rows for transformedTissuePositionList
                transformedTissuePositionList.append(row)
                
    registeredData['visiumTransformed'] = synXfm["warpedmovout"].numpy()
    registeredData['filteredFeatureMatrixGeneList'] = processedVisium['filteredFeatureMatrixGeneList']
    registeredData['geneListMasked'] = processedVisium['geneListMasked']
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
    tempDenseMatrix = processedVisium['filteredFeatureMatrixLog2'].todense()
    registeredData['filteredFeatureMatrixMasked'] = sp_sparse.csc_matrix(tempDenseMatrix[:,filteredFeatureMatrixMaskedIdx])
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
    type_of_transform='SyNAggro', grad_step=0.1, reg_iterations=(120, 100,80,60,40,20,0), \
    syn_sampling=32, flow_sigma=3, syn_metric='mattes', outprefix=os.path.join(processedVisium['derivativesPath'],f"{processedVisium['sampleID']}_to_{sampleToRegisterTo['sampleID']}_xfm"))
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
    registeredData['geneListMasked'] = processedVisium['geneListMasked']

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
    templateRegisteredData['geneListMasked'] = registeredVisium['geneListMasked']

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
    tempDenseMatrix = registeredVisium['filteredFeatureMatrixLog2'].todense()
    templateRegisteredData['filteredFeatureMatrixMasked'] = sp_sparse.csr_matrix(tempDenseMatrix[:,filteredFeatureMatrixMaskedIdx])
    #sp_sparse.save_npz(f"{os.path.join(registeredVisium['derivativesPath'],registeredVisium['sampleID'])}_OrderedLog2FeatureMatrixAllenTemplateMasked.npz", sp_sparse.csc_matrix(templateRegisteredData['filteredFeatureMatrixMasked']))
    cv2.imwrite(f"{os.path.join(registeredVisium['derivativesPath'],registeredVisium['sampleID'])}_registered_to_{bestSampleRegisteredToTemplate['sampleID']}_to_Allen.png",templateRegisteredData['visiumTransformed'])

    return templateRegisteredData

# create digital spots for an allen template slice
def createDigitalSpots(templateData, desiredSpotSize):
    w = np.sqrt(3) * (desiredSpotSize/2)   # width of pointy up hexagon
    h = desiredSpotSize    # height of pointy up hexagon
    currentX = 0
    currentY = 0
    rowCount = 0
    templateSpots = []
    while currentY < templateData['leftHem'].shape[0]:
        if currentX < templateData['leftHem'].shape[1]:
            templateSpots.append([currentX, currentY])
            currentX += w
        elif (currentX > templateData['leftHem'].shape[1]):
            rowCount += 1
            currentY += h * (3/4)
            if ((currentY < templateData['leftHem'].shape[0]) and (rowCount % 2)):
                currentX = w/2
            else:
                currentX = 0
        elif ((currentX > templateData['leftHem'].shape[1] * 10) and (currentY > templateData['leftHem'].shape[0] * 10)):
            print("something is wrong")

    templateSpots = np.array(templateSpots)

    # remove non-tissue spots
    roundedTemplateSpots = np.array(templateSpots.round(), dtype=int)
    ### the following line is dependent on bestSampleToTemplate, so either fix dependency or make input be bestSampleToTemplate
    digitalSpots = []
    for row in range(len(roundedTemplateSpots)):
        if bestSampleToTemplate['visiumTransformed'][roundedTemplateSpots[row,1],roundedTemplateSpots[row,0]] > 0:
            digitalSpots.append(templateSpots[row])
            
    digitalSpots = np.array(digitalSpots)
    # uncomment following 3 lines to see the digital template spots
    plt.imshow(templateData['leftHem'])
    plt.scatter(digitalSpots[:,0],digitalSpots[:,1], alpha=0.3)
    plt.show()
    return digitalSpots

# find nearest neighbor in digital allen spots for each sample spot
# kNN assuming 1 spot with 6 neighbors
def findDigitalNearestNeighbors(digitalSpots, templateRegisteredSpots, kNN):
    # finds distance between current spot and list
    allSpotNN = []
    allMeanCdists = []
    for actSpot in digitalSpots:
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

# read gene list from txt or csv file
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
