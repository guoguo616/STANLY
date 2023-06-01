#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""".
Created on Sep 7 2022

@author: zjpeters.
"""
# data being read in includes: json, h5, csv, nrrd, jpg, and svg
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
import time
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
    if os.path.exists(os.path.join(sampleFolder,"spatial")):
        spatialFolder = os.path.join(sampleFolder,"spatial")
        try:
            os.path.isfile(glob(os.path.join(sampleFolder, '*filtered_feature_bc_matrix.h5'))[0])
            dataFolder = sampleFolder
        except IndexError:
            os.path.isfile(glob(os.path.join(spatialFolder, '*filtered_feature_bc_matrix.h5'))[0])
            dataFolder = spatialFolder
        # dataFolder = os.path.join(sampleFolder)
    elif os.path.exists(os.path.join(sampleFolder,"outs","spatial")):
        spatialFolder = os.path.join(sampleFolder,"outs","spatial")
        try:
            os.path.isfile(glob(os.path.join(sampleFolder,"outs", '*filtered_feature_bc_matrix.h5'))[0])
            dataFolder = os.path.join(sampleFolder,"outs")
        except IndexError:
            os.path.isfile(glob(os.path.join(spatialFolder, '*filtered_feature_bc_matrix.h5'))[0])
            dataFolder = spatialFolder
        # dataFolder = os.path.join(sampleFolder,"outs")
    else:
        print("Something isn't working!")
    
    visiumData['imageData'] = io.imread(os.path.join(spatialFolder,"tissue_hires_image.png"))
    # visiumData['imageDataGray'] = 1 - visiumData['imageData'][:,:,2]
    visiumData['imageDataGray'] = 1 - color.rgb2gray(visiumData['imageData'])
    visiumData['sampleID'] = sampleFolder.rsplit(sep='/',maxsplit=1)[-1]
    tissuePositionsList = []
    tissueSpotBarcodes = []
    with open(os.path.join(spatialFolder,"tissue_positions_list.csv"), newline='') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',')
        for row in csvreader:
            # checks for in tissue spots
            if row[1] == '1':
                tissueSpotBarcodes.append(row[0])
                tissuePositionsList.append(row[1:])
    tissuePositionsList = np.array(tissuePositionsList, dtype=float)
    visiumData['tissueSpotBarcodeList'] = tissueSpotBarcodes
    visiumData['tissuePositionsList'] = tissuePositionsList
    scaleFactorPath = open(os.path.join(spatialFolder,"scalefactors_json.json"))
    visiumData['scaleFactors'] = json.loads(scaleFactorPath.read())
    scaleFactorPath.close()
    filteredFeatureMatrixPath = glob(os.path.join(dataFolder,"*filtered_feature_bc_matrix.h5"))
    filteredFeatureMatrix = get_matrix_from_h5(os.path.join(filteredFeatureMatrixPath[0]))
    visiumData['filteredFeatureMatrix'] = filteredFeatureMatrix
    # the ratio of real spot diameter, 55um, by imaged resolution of spot
    visiumData['spotStartingResolution'] = 0.55 / visiumData["scaleFactors"]["spot_diameter_fullres"]
    # plt.imshow(visiumData['imageData'])
    return visiumData

# use to select which allen slice to align visium data to and import relevant data
def chooseTemplateSlice(sliceLocation):
    ccfPath = os.path.join(refDataPath,'data','ccf')
    # checks if ccf data has been downloaded already and downloads it if it hasn't
    if not os.path.exists(ccfPath):
        print("Downloading 10 micron resolution ara_nissl nrrd file from the Allen Institute, this may take awhile")
        os.mkdirs(ccfPath)
        from allensdk.core.reference_space_cache import ReferenceSpaceApi
        rsapi = ReferenceSpaceApi()
        rsapi.download_volumetric_data('ara_nissl','ara_nissl_10.nrrd',10, save_file_path=os.path.join(ccfPath, 'ara_nissl_10.nrrd'))
    ara_data = ants.image_read(os.path.join(ccfPath,'ara_nissl_10.nrrd'))
        
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
    templateData['leftHemAnnot'] = np.array(templateAnnotationLeft, dtype='int')
    templateData['rightHemAnnot'] = np.array(templateAnnotationRight, dtype='int')
    annotX = templateLeft.shape[0]
    annotY = templateLeft.shape[1]
    templateAnnotLeftRGB = np.zeros([annotX, annotY, 3])
    templateAnnotLeftUnique = np.unique(templateData['leftHemAnnot'])
    templateAnnotLeftRenum = np.zeros(templateData['leftHemAnnot'].shape)
    for newNum, origNum in enumerate(templateAnnotLeftUnique):
        templateAnnotLeftRenum[templateData['leftHemAnnot'] == origNum] = newNum
    # templateAnnotLeftRGB[:,:,1] = templateAnnotLeftRGB[:,:,1] - morphology.binary_dilation(feature.canny(templateAnnotLeftRenum))
    # templateData['leftHemAnnotEdges']  = templateAnnotLeftRGB
    se = morphology.disk(2)
    templateAnnotLeftRGB = morphology.binary_dilation(feature.canny(templateAnnotLeftRenum), footprint=se)
    templateData['leftHemAnnotEdges']  = templateAnnotLeftRGB
    # templateData['leftHemAnnotEdges'] = morphology.binary_dilation(feature.canny(templateAnnotLeftRenum))
    templateAnnotRightUnique = np.unique(templateData['rightHemAnnot'])
    templateAnnotRightRenum = np.zeros(templateData['rightHemAnnot'].shape)
    for newNum, origNum in enumerate(templateAnnotRightUnique):
        templateAnnotRightRenum[templateData['rightHemAnnot'] == origNum] = newNum
    templateData['rightHemAnnotEdges'] = feature.canny(templateAnnotRightRenum)*255
    # currently using the 10um resolution atlas, would need to change if that changes
    templateData['startingResolution'] = 0.01
    annotation_id = []
    annotation_name = []
    structure_id_path = []
    with open(os.path.join(codePath,'data','allen_ccf_annotation.csv'), newline='') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',')
        next(csvreader)
        for row in csvreader:
            # imports the annotation id, and full name
            annotation_id.append(row[3])
            annotation_name.append(row[4])
            structure_id_path.append(row[7])

    templateData['annotationID'] = annotation_id
    templateData['annotationName'] = annotation_name
    templateData['structureIDPath'] = structure_id_path
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
        # tissuePointsResizeRotate[:,0] = tissuePointsResizeRotate[:,0]
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
    elif rotation == -180:
        rotMat = [[1,0],[0,-1]]
        tissuePointsResizeRotate = np.matmul(tissuePointsResizeToHighRes, rotMat)
        # tissuePointsResizeRotate[:,0] = tissuePointsResizeRotate[:,0] #+ visiumData["imageDataGray"].shape[1]
        tissuePointsResizeRotate[:,1] = tissuePointsResizeRotate[:,1] + visiumData["imageDataGray"].shape[0]
    else:
        print("Incorrect rotation! Please enter: 0, 90, 180, or 270")
        print("To flip image across axis, use a - before the rotation, i.e. -180 to rotate an image 180 degrees and flip across hemisphere")
    
    return tissuePointsResizeRotate

# prepares visium data for registration
def processVisiumData(visiumData, templateData, rotation):
    processedVisium = {}
    # the sampleID might have issues on non unix given the slash direction, might need to fix
    processedVisium['sampleID'] = visiumData['sampleID']
    outputPath = os.path.join(derivatives, visiumData['sampleID'])
    try:
        file = open(f"{os.path.join(outputPath,processedVisium['sampleID'])}_tissuePointsProcessed.csv", 'r')
        print(f"{processedVisium['sampleID']} has already been processed! Loading data")
        processedVisium = loadProcessedSample(outputPath)
        return processedVisium
    except IOError:
        print(f"Processing {processedVisium['sampleID']}")
    if not os.path.exists(outputPath):
        os.makedirs(outputPath)
    resolutionRatio = visiumData['spotStartingResolution'] / templateData['startingResolution']
    processedVisium['derivativesPath'] = outputPath
    # processedVisium['tissueSpotBarcodeList'] = visiumData['tissueSpotBarcodeList']
    # processedVisium['degreesOfRotation'] = int(rotation)
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
    # processedVisium['resolutionRatio'] = visiumData['spotStartingResolution'] / templateData['startingResolution']
    tissueResized = rescale(tissueNormalized,resolutionRatio)
    if rotation < 0:
        tissuePointsRotated = rotateTissuePoints(visiumData, rotation)
        rotation = rotation * -1
        tissueRotated = rotate(tissueResized, rotation, resize=True)
        tissueRotated = tissueRotated[:,::-1]
    else:
        tissueRotated = rotate(tissueResized, rotation, resize=True)
        tissuePointsRotated = rotateTissuePoints(visiumData, rotation)
    processedVisium['tissueProcessed'] = match_histograms(tissueRotated, templateData['leftHem'])
    processedVisium['tissueProcessed'] = processedVisium['tissueProcessed'] - processedVisium['tissueProcessed'].min()
    
    tissuePointsResized = tissuePointsRotated * resolutionRatio
    # processedVisium['tissuePointsResizedForTransform'] = processedVisium['tissuePointsRotated'] * processedVisium['resolutionRatio']
    # processedVisium['tissuePointsResizedForTransform'][:,[0,1]] = processedVisium['tissuePointsResizedForTransform'][:,[1,0]]
    
    filteredFeatureMatrixGeneList = []
    for geneName in visiumData['filteredFeatureMatrix'][0]['name']:
        filteredFeatureMatrixGeneList.append(geneName.decode())
    # processedVisium['filteredFeatureMatrixGeneList'] = filteredFeatureMatrixGeneList

    # this orders the filtered feature matrix so that the columns are in the order of the coordinate list, so barcodes no longer necessary
    filteredFeatureMatrixBarcodeList = []
    for barcodeID in visiumData['filteredFeatureMatrix'][1]:
        filteredFeatureMatrixBarcodeList.append(barcodeID.decode())
    # processedVisium['filteredFeatureMatrixBarcodeList'] = filteredFeatureMatrixBarcodeList 

    tissueSpotBarcodeListSorted = []
    for actbarcode in visiumData['tissueSpotBarcodeList']:
        tissueSpotBarcodeListSorted.append(filteredFeatureMatrixBarcodeList.index(actbarcode))
    # no need to keep the dense version in the processedVisium dictionary since the necessary information is in the ordered matrix and coordinates
    # processedVisium['tissueSpotBarcodeListSorted'] = tissueSpotBarcodeListSorted
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
    geneMask = countsPerGene > 0
    geneMask = np.squeeze(np.array(geneMask))    
    filteredFeatureMatrixGeneList = np.array(filteredFeatureMatrixGeneList)
    processedVisium['geneListMasked'] = filteredFeatureMatrixGeneList[geneMask].tolist()
    orderedDenseMatrixSpotMasked = orderedDenseMatrix[:,spotMask]
    orderedDenseMatrixSpotMasked = orderedDenseMatrixSpotMasked[geneMask,:]
    processedVisium['spotCount'] = orderedDenseMatrixSpotMasked.shape[1]
    print(f"{processedVisium['sampleID']} has {processedVisium['spotCount']} spots")
    processedVisium['processedTissuePositionList'] = tissuePointsResized[spotMask,:]
    # processedVisium['filteredFeatureMatrixOrdered'] = sp_sparse.csc_matrix(orderedDenseMatrixSpotMasked)
    processedVisium['filteredFeatureMatrixLog2'] = sp_sparse.csc_matrix(np.log2((orderedDenseMatrixSpotMasked + 1)))
    
    finiteMin=np.min(countsPerSpotZscore)
    finiteMax=np.max(countsPerSpotZscore)
    zeroCenteredCmap = mcolors.TwoSlopeNorm(0,vmin=finiteMin, vmax=finiteMax)
    # plt.imshow( processedVisium['tissueRotated'], cmap='gray')
    # plt.scatter(processedVisium['tissuePointsResized'][:,0],processedVisium['tissuePointsResized'][:,1], c=np.array(countsPerSpot), alpha=0.8)
    # plt.title(f"Total gene count per spot for {processedVisium['sampleID']}")
    # plt.colorbar()
    # plt.show()
    plt.imshow(processedVisium['tissueProcessed'], cmap='gray')
    plt.scatter(tissuePointsResized[:,0],tissuePointsResized[:,1], c=np.array(countsPerSpotZscore), alpha=0.8, cmap='seismic', norm=zeroCenteredCmap, marker='.')
    plt.title(f"Z-score of overall gene count per spot for {processedVisium['sampleID']}")
    plt.colorbar()
    plt.show()
    
    # write outputs
    # writes json containing general info and masked gene list
    processedDataDict = {
        "sampleID": visiumData['sampleID'],
        "rotation": int(rotation),
        "resolutionRatio": visiumData['spotStartingResolution'] / templateData['startingResolution'],
        "spotCount": processedVisium['spotCount'],
        "otsuThreshold": float(otsuThreshold),
        "geneList": processedVisium['geneListMasked']
    }
        # Serializing json
    json_object = json.dumps(processedDataDict, indent=4)
     
    # Writing to sample.json
    with open(f"{processedVisium['derivativesPath']}/{processedVisium['sampleID']}_processing_information.json", "w") as outfile:
        outfile.write(json_object)
    # writes sorted, masked, normalized filtered feature matrix to .npz file
    sp_sparse.save_npz(f"{os.path.join(outputPath,processedVisium['sampleID'])}_tissuePointOrderedFeatureMatrix.npz", processedVisium['filteredFeatureMatrixLog2'])
    
    # writes image for masked greyscale tissue, as well as the processed image that will be used in registration
    cv2.imwrite(f"{processedVisium['derivativesPath']}/{processedVisium['sampleID']}_tissue.png",255*tissue)
    cv2.imwrite(f"{processedVisium['derivativesPath']}/{processedVisium['sampleID']}_tissueProcessed.png",processedVisium['tissueProcessed'])
    
    header=['x','y','z','t','label','comment']
    rowFormat = []
    # the x and y are swapped between ants and numpy, but this is so far dealt with within the code
    with open(f"{os.path.join(outputPath,processedVisium['sampleID'])}_tissuePointsProcessed.csv", 'w', encoding='UTF8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i in range(len(processedVisium['processedTissuePositionList'])):
            rowFormat = [processedVisium['processedTissuePositionList'][i,1]] + [processedVisium['processedTissuePositionList'][i,0]] + [0] + [0] + [0] + [0]
            writer.writerow(rowFormat)
    return processedVisium


def runANTsToAllenRegistration(processedVisium, templateData):
    # registeredData will contain: sampleID, derivativesPath, transformedTissuePositionList, fwdtransforms, invtransforms
    registeredData = {}
    try:
        file = open(f"{os.path.join(processedVisium['derivativesPath'],processedVisium['sampleID'])}_tissuePointsProcessedToAllen.csv", 'r')
        print(f"{processedVisium['sampleID']} has already been processed! check {processedVisium['derivativesPath']}")
        print(f"Loading data for {processedVisium['sampleID']}")
        registeredData['sampleID'] = processedVisium['sampleID']
        registeredData['derivativesPath'] = processedVisium['derivativesPath']
        registeredData['fwdtransforms'] = [os.path.join(processedVisium['derivativesPath'],f"{processedVisium['sampleID']}_xfm1Warp.nii.gz"),os.path.join(processedVisium['derivativesPath'],f"{processedVisium['sampleID']}_xfm0GenericAffine.mat")]
        registeredData['invtransforms'] = [os.path.join(processedVisium['derivativesPath'],f"{processedVisium['sampleID']}_xfm0GenericAffine.mat"), os.path.join(processedVisium['derivativesPath'],f"{processedVisium['sampleID']}_xfm1InverseWarp.nii.gz"),]
        registeredData['visiumTransformed'] = plt.imread(os.path.join(processedVisium['derivativesPath'],f"{processedVisium['sampleID']}_tissue_registered_to_Allen_slice_{templateData['sliceNumber']}.png"))
        return
    except IOError:
        print(f"Registering {processedVisium['sampleID']}")
        # convert into ants image type
        registeredData['sampleID'] = processedVisium['sampleID']
        registeredData['derivativesPath'] = processedVisium['derivativesPath']
        templateAntsImage = ants.from_numpy(templateData['leftHem'])
        sampleAntsImage = ants.from_numpy(processedVisium['tissueProcessed'])
        # run registration
        synXfm = ants.registration(fixed=templateAntsImage, moving=sampleAntsImage, \
        type_of_transform='SyNAggro', grad_step=0.1, reg_iterations=(120, 100,80,60,40,20,0), \
        syn_sampling=32, flow_sigma=3,syn_metric='mattes', outprefix=os.path.join(processedVisium['derivativesPath'],f"{processedVisium['sampleID']}_xfm"))
        
        registeredData['antsOutput'] = synXfm
        registeredData['fwdtransforms'] = synXfm['fwdtransforms']
        registeredData['invtransforms'] = synXfm['invtransforms']

        # apply syn transform to tissue spot coordinates
        applyTransformStr = f"antsApplyTransformsToPoints -d 2 -i {os.path.join(processedVisium['derivativesPath'],processedVisium['sampleID'])}_tissuePointsProcessed.csv -o {os.path.join(processedVisium['derivativesPath'],processedVisium['sampleID'])}_tissuePointsProcessedToAllen.csv -t [ {os.path.join(processedVisium['derivativesPath'],processedVisium['sampleID'])}_xfm0GenericAffine.mat,1] -t [{os.path.join(processedVisium['derivativesPath'],processedVisium['sampleID'])}_xfm1InverseWarp.nii.gz]"
        pid = os.system(applyTransformStr)
        # program has to wait while spots are transformed by the system
        if pid:
            os.wait()
        #     print("Applying transformation to spots")
        # else:
        #     print("Finished transforming spots!")
    
        registeredData['visiumTransformed'] = synXfm["warpedmovout"].numpy()
        # registeredData['filteredFeatureMatrixGeneList'] = processedVisium['filteredFeatureMatrixGeneList']
        registeredData['geneListMasked'] = processedVisium['geneListMasked']
    
    transformedTissuePositionList = []
    with open(os.path.join(f"{os.path.join(processedVisium['derivativesPath'],processedVisium['sampleID'])}_tissuePointsProcessedToAllen.csv"), newline='') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',')
            next(csvreader)
            for row in csvreader:
                #####################
                # should be able to fix with:
                # transformedTissuePositionList.append(row[1,0])
                # then remove the two rows for transformedTissuePositionList
                transformedTissuePositionList.append(row)
                

    transformedTissuePositionList = np.array(transformedTissuePositionList, dtype=float)
    # switching x,y columns back to python compatible and deleting empty columns
    transformedTissuePositionList[:,[0,1]] = transformedTissuePositionList[:,[1,0]]
    registeredData['transformedTissuePositionList'] = np.delete(transformedTissuePositionList, [2,3,4,5],1)

    # plt.imshow(registeredData['visiumTransformed'])
    # plt.scatter(registeredData['transformedTissuePositionList'][0:,0],registeredData['transformedTissuePositionList'][0:,1], marker='.', c='red', alpha=0.3)
    # plt.show()
    
    # plt.imshow(registeredData['visiumTransformed'],cmap='gray')
    # plt.imshow(templateData['leftHem'], alpha=0.3)
    # plt.title(processedVisium['sampleID'])
    # plt.show()
        
    transformedTissuePositionListMask = np.logical_and(registeredData['transformedTissuePositionList'] > 0, registeredData['transformedTissuePositionList'] < registeredData['visiumTransformed'].shape[0])
    maskedTissuePositionList = []
    # filteredFeatureMatrixBinaryMask = []
    # filteredFeatureMatrixMasked = np.zeros(processedVisium['filteredFeatureMatrixOrdered'][:,0].shape)
    filteredFeatureMatrixMaskedIdx = []
    for i, masked in enumerate(transformedTissuePositionListMask):
        if masked.all() == True:
            filteredFeatureMatrixMaskedIdx.append(i)
            # filteredFeatureMatrixBinaryMask.append(1)
            maskedTissuePositionList.append(registeredData['transformedTissuePositionList'][i])
            # filteredFeatureMatrixMasked = np.append(filteredFeatureMatrixMasked, processedVisium['filteredFeatureMatrixOrdered'][:,i],axis=1)
        # else:
            # filteredFeatureMatrixBinaryMask.append(0)
    registeredData['maskedTissuePositionList'] = np.array(maskedTissuePositionList, dtype=float)

    # registeredData['filteredFeatureMatrixMasked'] = np.delete(filteredFeatureMatrixMasked, 0,1)
    tempDenseMatrix = processedVisium['filteredFeatureMatrixLog2'].todense()
    registeredData['filteredFeatureMatrixMasked'] = sp_sparse.csc_matrix(tempDenseMatrix[:,filteredFeatureMatrixMaskedIdx])
    # write re-ordered filtered feature matrix csv to match tissue spot order
    sp_sparse.save_npz(f"{os.path.join(processedVisium['derivativesPath'],processedVisium['sampleID'])}_OrderedLog2FeatureMatrixTemplateMasked.npz", sp_sparse.csc_matrix(registeredData['filteredFeatureMatrixMasked']))        
    cv2.imwrite(f"{registeredData['derivativesPath']}/{registeredData['sampleID']}_tissue_registered_to_Allen_slice_{templateData['sliceNumber']}.png",registeredData['visiumTransformed'])
    
    return registeredData

def runANTsInterSampleRegistration(processedVisium, sampleToRegisterTo):
    # convert into ants image type
    registeredData = {}
    templateAntsImage = ants.from_numpy(sampleToRegisterTo['tissueProcessed'])
    sampleAntsImage = ants.from_numpy(processedVisium['tissueProcessed'])
    # mattes seems to be most conservative syn_metric
    synXfm = ants.registration(fixed=templateAntsImage, moving=sampleAntsImage, \
    type_of_transform='SyNAggro', grad_step=0.1, reg_iterations=(120, 100,80,60,40,20,0), \
    syn_sampling=32, flow_sigma=3, syn_metric='mattes', outprefix=os.path.join(processedVisium['derivativesPath'],f"{processedVisium['sampleID']}_to_{sampleToRegisterTo['sampleID']}_xfm"))
    registeredData['antsOutput'] = synXfm
    registeredData['sampleID'] = processedVisium['sampleID']
    registeredData['derivativesPath'] = processedVisium['derivativesPath']
    applyTransformStr = f"antsApplyTransformsToPoints -d 2 -i {os.path.join(processedVisium['derivativesPath'],processedVisium['sampleID'])}_tissuePointsProcessed.csv -o {os.path.join(processedVisium['derivativesPath'],processedVisium['sampleID'])}_tissuePointsResize_to_{sampleToRegisterTo['sampleID']}TransformApplied.csv -t [ {os.path.join(processedVisium['derivativesPath'],processedVisium['sampleID'])}_to_{sampleToRegisterTo['sampleID']}_xfm0GenericAffine.mat,1] -t {os.path.join(processedVisium['derivativesPath'],processedVisium['sampleID'])}_to_{sampleToRegisterTo['sampleID']}_xfm1InverseWarp.nii.gz"
    # program has to wait while the transformation is applied by the system
    pid = os.system(applyTransformStr)
    if pid:
        os.wait()
    #     print("Applying transformation to spots")
    # else:
    #     print("Finished transforming spots!")
    transformedTissuePositionList = []
    with open(os.path.join(f"{os.path.join(processedVisium['derivativesPath'],processedVisium['sampleID'])}_tissuePointsResize_to_{sampleToRegisterTo['sampleID']}TransformApplied.csv"), newline='') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',')
            next(csvreader)
            for row in csvreader:
                transformedTissuePositionList.append(row)
                
    registeredData['visiumTransformed'] = synXfm["warpedmovout"].numpy()
    # registeredData['filteredFeatureMatrixGeneList'] = processedVisium['filteredFeatureMatrixGeneList']
    registeredData['geneListMasked'] = processedVisium['geneListMasked']

    registeredData['transformedTissuePositionList'] = np.array(transformedTissuePositionList, dtype=float)
    # switching x,y columns back to python compatible and deleting empty columns
    registeredData['transformedTissuePositionList'][:,[0,1]] = registeredData['transformedTissuePositionList'][:,[1,0]]
    registeredData['transformedTissuePositionList'] = np.delete(registeredData['transformedTissuePositionList'], [2,3,4,5],1)
    # registeredData['tissueSpotBarcodeList'] = processedVisium['tissueSpotBarcodeList']
    registeredData['filteredFeatureMatrixLog2'] = processedVisium['filteredFeatureMatrixLog2']
    plt.imshow(registeredData['visiumTransformed'], cmap='gray')
    plt.scatter(registeredData['transformedTissuePositionList'][0:,0],registeredData['transformedTissuePositionList'][0:,1], marker='.', c='red', alpha=0.3)
    plt.show()
    
    plt.imshow(sampleToRegisterTo['tissueProcessed'], cmap='gray')
    plt.imshow(registeredData['visiumTransformed'], alpha=0.7, cmap='gray')
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
                
    templateRegisteredData['derivativesPath'] = registeredVisium['derivativesPath']
    templateRegisteredData['sampleID'] = registeredVisium['sampleID']
    templateRegisteredData['bestFitSampleID'] = bestSampleRegisteredToTemplate['sampleID']
    templateRegisteredData['visiumTransformed'] = sampleToTemplate.numpy()

    transformedTissuePositionList = np.array(transformedTissuePositionList, dtype=float)
    # switching x,y columns back to python compatible and deleting empty columns
    transformedTissuePositionList[:,[0,1]] = transformedTissuePositionList[:,[1,0]]
    transformedTissuePositionList = np.delete(transformedTissuePositionList, [2,3,4,5],1)
    templateRegisteredData['geneListMasked'] = registeredVisium['geneListMasked']

    plt.imshow(templateRegisteredData['visiumTransformed'], cmap='gray')
    plt.scatter(transformedTissuePositionList[0:,0],transformedTissuePositionList[0:,1], marker='.', c='red', alpha=0.3)
    plt.show()

    plt.imshow(templateData['leftHem'], cmap='gray')    
    plt.imshow(templateRegisteredData['visiumTransformed'],alpha=0.8,cmap='gray')
    plt.title(templateRegisteredData['sampleID'])
    plt.show()
        
    transformedTissuePositionListMask = np.logical_and(transformedTissuePositionList > 0, transformedTissuePositionList < templateRegisteredData['visiumTransformed'].shape[0])
    maskedTissuePositionList = []
    filteredFeatureMatrixMaskedIdx = []
    for i, masked in enumerate(transformedTissuePositionListMask):
        if masked.all() == True:
            filteredFeatureMatrixMaskedIdx.append(i)
            maskedTissuePositionList.append(transformedTissuePositionList[i])
            # filteredFeatureMatrixMasked = np.append(filteredFeatureMatrixMasked, registeredVisium['filteredFeatureMatrixOrdered'][:,i],axis=1)
    templateRegisteredData['maskedTissuePositionList'] = np.array(maskedTissuePositionList, dtype=float)
    tempDenseMatrix = registeredVisium['filteredFeatureMatrixLog2'].todense()
    templateRegisteredData['filteredFeatureMatrixMasked'] = sp_sparse.csr_matrix(tempDenseMatrix[:,filteredFeatureMatrixMaskedIdx])
    # imageFilename = f"os.path.join({registeredVisium['derivativesPath']},{registeredVisium['sampleID'])}_registered_to_{bestSampleRegisteredToTemplate['sampleID']}_to_Allen.png"
    imageFilename = os.path.join(registeredVisium['derivativesPath'],f"{registeredVisium['sampleID']}_registered_to_{bestSampleRegisteredToTemplate['sampleID']}_to_Allen.png")
    cv2.imwrite(imageFilename, templateRegisteredData['visiumTransformed'])

    return templateRegisteredData

# create digital spots for an allen template slice
def createDigitalSpots(templateRegisteredData, desiredSpotSize):
    w = np.sqrt(3) * (desiredSpotSize/2)   # width of pointy up hexagon
    h = desiredSpotSize    # height of pointy up hexagon
    currentX = 0
    currentY = 0
    rowCount = 0
    templateSpots = []
    while currentY < templateRegisteredData['visiumTransformed'].shape[0]:
        if currentX < templateRegisteredData['visiumTransformed'].shape[1]:
            templateSpots.append([currentX, currentY])
            currentX += w
        elif (currentX > templateRegisteredData['visiumTransformed'].shape[1]):
            rowCount += 1
            currentY += h * (3/4)
            if ((currentY < templateRegisteredData['visiumTransformed'].shape[0]) and (rowCount % 2)):
                currentX = w/2
            else:
                currentX = 0
        elif ((currentX > templateRegisteredData['visiumTransformed'].shape[1] * 10) and (currentY > templateRegisteredData['visiumTransformed'].shape[0] * 10)):
            print("something is wrong")

    templateSpots = np.array(templateSpots)

    # remove non-tissue spots
    roundedTemplateSpots = np.array(templateSpots.round(), dtype=int)
    ### the following line is dependent on bestSampleToTemplate, so either fix dependency or make input be bestSampleToTemplate
    digitalSpots = []
    for row in range(len(roundedTemplateSpots)):
        if templateRegisteredData['visiumTransformed'][roundedTemplateSpots[row,1],roundedTemplateSpots[row,0]] > 0:
            digitalSpots.append(templateSpots[row])
            
    digitalSpots = np.array(digitalSpots)
    # uncomment following 3 lines to see the digital template spots
    plt.imshow(templateRegisteredData['visiumTransformed'])
    plt.scatter(digitalSpots[:,0],digitalSpots[:,1], alpha=0.3)
    plt.show()
    # write csv of digital spots in ants format
    derivativesPath = templateRegisteredData['derivativesPath'].split('/')
    del derivativesPath[-1]
    derivativesPath = os.path.join('/',*derivativesPath)
    with open(os.path.join(derivativesPath,'digitalSpotCoordinates.csv'), 'w', encoding='UTF8') as f:
        header=['x','y','z','t','label','comment']
        writer = csv.writer(f)
        writer.writerow(header)
        for i in range(len(digitalSpots)):
            rowFormat = [digitalSpots[i,1]] + [digitalSpots[i,0]] + [0] + [0] + [0] + [0]
            writer.writerow(rowFormat)
    return digitalSpots

# find nearest neighbor in digital allen spots for each sample spot
# kNN assuming 1 spot with 6 neighbors
def findDigitalNearestNeighbors(digitalSpots, templateRegisteredSpots, kNN, spotDist):
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
        blankIdx[:] = -9999
        spotNNIdx = []
        for i in actSpotCdist:
            if spotMeanCdist < (spotDist * 3):
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
# it seems that reading from the csv is probably faster than searching the allen sdk, but will time it at some point
def createRegionalMask(template, desiredRegion):
    regionIdx = template['annotationName'].index(desiredRegion)
    regionID = int(template['annotationID'][regionIdx])
    # regionStructs = template['structureIDPath']
    structIDs = []
    for idx, actStruct in enumerate(np.array(template['structureIDPath'])):
        if f"/{regionID}/" in actStruct:
            structIDs.append(template['annotationID'][idx])
    structIDs = np.array(structIDs)
    regionMask = np.zeros(template['leftHemAnnot'].shape, dtype='int')  
    # maskList = []
    for actID in structIDs:
        # structMask = np.zeros(template['leftHemAnnot'].shape, dtype='int')        
        regionBoolMask = np.where(template['leftHemAnnot'] == int(actID))
        if any(regionBoolMask[0]):
            regionMask[regionBoolMask[0],regionBoolMask[1]] = 1
        # regionMask = regionMask + structMask
        
    plt.imshow(template['leftHem'], cmap='gray')
    plt.imshow(regionMask, alpha=0.8)
    plt.show()
    return regionMask

def createRegionMaskAllenSDK(template, desiredRegion):
    from allensdk.core.reference_space_cache import ReferenceSpaceCache
    reference_space_key = 'annotation/ccf_2017'
    resolution = 10
    rspc = ReferenceSpaceCache(resolution, reference_space_key, manifest='manifest.json')
    rsp = rspc.get_reference_space()
    # structure graph id is 1 for mouse
    tree = rspc.get_structure_tree(structure_graph_id=1) 
    # regionList = tree.get_name_map()
    region = tree.get_structures_by_name([desiredRegion])
    fullMask = rsp.make_structure_mask([region[0]['id']])
    # the 570 here is to split the mask in half
    regionMask = fullMask[(template['sliceNumber'] * resolution),:,570:]
    return regionMask

def createRegionalDigitalSpots(regionMask, desiredSpotSize):
    w = np.sqrt(3) * (desiredSpotSize/2)   # width of pointy up hexagon
    h = desiredSpotSize    # height of pointy up hexagon
    currentX = 0
    currentY = 0
    rowCount = 0
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
    roundedTemplateSpots = np.array(templateSpots.round(), dtype=int)
    ### the following line is dependent on bestSampleToTemplate, so either fix dependency or make input be bestSampleToTemplate
    digitalSpots = []
    for row in range(len(roundedTemplateSpots)):
        if regionMask[roundedTemplateSpots[row,1],roundedTemplateSpots[row,0]] > 0:
            digitalSpots.append(templateSpots[row])
            
    digitalSpots = np.array(digitalSpots)
    # uncomment following 3 lines to see the digital template spots
    # plt.imshow(regionMask)
    # plt.scatter(digitalSpots[:,0],digitalSpots[:,1], alpha=0.3)
    # plt.show()
    return digitalSpots

def loadProcessedSample(locOfProcessedSample):
    processedVisium = {}
    processedVisium['derivativesPath'] = os.path.join(locOfProcessedSample)
    processedVisium['sampleID'] = locOfProcessedSample.rsplit(sep='/',maxsplit=1)[-1]
    processedVisium['tissueProcessed'] = io.imread(os.path.join(processedVisium['derivativesPath'],f"{processedVisium['sampleID']}_tissueProcessed.png"))
    jsonPath = open(os.path.join(processedVisium['derivativesPath'],f"{processedVisium['sampleID']}_processing_information.json"))
    processedSampleJson = json.loads(jsonPath.read())
    processedVisium['geneListMasked'] = processedSampleJson['geneList']
    processedVisium['spotCount'] = processedSampleJson['spotCount']
    filteredFeatureMatrixLog2 = sp_sparse.load_npz(os.path.join(processedVisium['derivativesPath'], f"{processedVisium['sampleID']}_tissuePointOrderedFeatureMatrix.npz"))
    processedVisium['filteredFeatureMatrixLog2'] = filteredFeatureMatrixLog2
    tissuePositionList = []
    with open(os.path.join(f"{os.path.join(processedVisium['derivativesPath'],processedVisium['sampleID'])}_tissuePointsProcessed.csv"), newline='') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',')
            next(csvreader)
            for row in csvreader:
                tissuePositionList.append(row)
                
    tissuePositionList = np.array(tissuePositionList, dtype=float)
    # switching x,y columns back to python compatible and deleting empty columns
    tissuePositionList[:,[0,1]] = tissuePositionList[:,[1,0]]
    processedVisium['processedTissuePositionList'] = np.delete(tissuePositionList, [2,3,4,5],1)
    return processedVisium

def loadAllenRegisteredSample(locOfRegSample):
    templateRegisteredData = {}
    templateRegisteredData['derivativesPath'] = os.path.join(locOfRegSample)
    templateRegisteredData['sampleID'] = locOfRegSample.rsplit(sep='/',maxsplit=1)[-1]
    try:
        bestFitSample = glob(os.path.join(locOfRegSample, f"{templateRegisteredData['sampleID']}_registered_to_*_to_Allen.png"))
        bestFitSample = bestFitSample[0]
    except IndexError:
        print(f"No registered data found in {locOfRegSample}")
    templateRegisteredData['visiumTransformed'] = io.imread(bestFitSample)
    bestFitSample = bestFitSample.rsplit(sep='/',maxsplit=1)[-1]
    # id of best fit is the third from the end
    bestFitSample = bestFitSample.rsplit(sep='_')[-3]
    templateRegisteredData['bestFitSampleID'] = bestFitSample
    jsonPath = open(os.path.join(templateRegisteredData['derivativesPath'],f"{templateRegisteredData['sampleID']}_processing_information.json"))
    processedSampleJson = json.loads(jsonPath.read())
    templateRegisteredData['geneListMasked'] = processedSampleJson['geneList']
    templateRegisteredData['spotCount'] = processedSampleJson['spotCount']
    transformedTissuePositionList = []
    with open(os.path.join(f"{os.path.join(templateRegisteredData['derivativesPath'],templateRegisteredData['sampleID'])}_tissuePointsResize_to_{templateRegisteredData['bestFitSampleID']}TemplateTransformApplied.csv"), newline='') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',')
            next(csvreader)
            for row in csvreader:
                transformedTissuePositionList.append(row)
    transformedTissuePositionList = np.array(transformedTissuePositionList, dtype=float)
    transformedTissuePositionList[:,[0,1]] = transformedTissuePositionList[:,[1,0]]
    transformedTissuePositionList = np.delete(transformedTissuePositionList, [2,3,4,5],1)
    transformedTissuePositionListMask = []
    transformedTissuePositionListMask = np.logical_and(transformedTissuePositionList > 0, transformedTissuePositionList < templateRegisteredData['visiumTransformed'].shape[0])
    maskedTissuePositionList = []
    filteredFeatureMatrixMaskedIdx = []
    for i, masked in enumerate(transformedTissuePositionListMask):
        if masked.all() == True:
            filteredFeatureMatrixMaskedIdx.append(i)
            maskedTissuePositionList.append(transformedTissuePositionList[i])
            # filteredFeatureMatrixMasked = np.append(filteredFeatureMatrixMasked, registeredVisium['filteredFeatureMatrixOrdered'][:,i],axis=1)
    templateRegisteredData['maskedTissuePositionList'] = np.array(maskedTissuePositionList, dtype=float)
    
    filteredFeatureMatrixLog2 = sp_sparse.load_npz(os.path.join(templateRegisteredData['derivativesPath'], f"{templateRegisteredData['sampleID']}_tissuePointOrderedFeatureMatrix.npz"))
    tempDenseMatrix = filteredFeatureMatrixLog2.todense()
    templateRegisteredData['filteredFeatureMatrixMasked'] = sp_sparse.csr_matrix(tempDenseMatrix[:,filteredFeatureMatrixMaskedIdx])
    return templateRegisteredData

def viewGeneInProcessedVisium(processedSample, geneName):
    try:
        geneIndex = processedSample['geneListMasked'].index(geneName)
        actSpots = processedSample['filteredFeatureMatrixLog2'][geneIndex, :]
        plt.imshow(processedSample['tissueProcessed'], cmap='gray')
        plt.scatter(processedSample['processedTissuePositionList'][:,0],processedSample['processedTissuePositionList'][:,1], c=np.array(actSpots.todense()), alpha=0.8, cmap='Reds', marker='.')
        plt.title(f'Gene count for {geneName} in {processedSample["sampleID"]}')
        plt.colorbar()
        # plt.savefig(os.path.join(derivatives,f'geneCount{geneName}{processedSample["sampleID"]}Registered.png'), bbox_inches='tight', dpi=300)
        plt.show()
    except(ValueError):
        print(f'{geneName} not found in dataset')

#%%
def runTTest(experiment, experimentalGroup, geneList, fdr='sidak', alpha=0.05):
    # needs to include:
    # benjamini-hochberg, bonferroni, and sidak fdr corrections
    # option to suppress image output, alternatively default off
    # experiment in this case is formatted as previous with each sample having `digitalSpotNearestNeighbors`
    # experimentalGroup is a binary list of [0,1] indicating whether or not sample is in experimental group
    nOfGenes = len(geneList)
    nDigitalSpots = experiment[0]['digitalSpotNearestNeighbors'].shape[0]
    nNearestNeighbors = experiment[0]['digitalSpotNearestNeighbors'].shape[1]
    if fdr == 'sidak':
        alphaFDR = 1 - np.power((1 - alpha),(1/(nOfGenes*nDigitalSpots)))
    elif fdr == 'benjamini-hochberg':
        rankList = np.arange(1,nDigitalSpots+1)
        alphaFDR = (rankList/(nOfGenes*nDigitalSpots)) * alpha
    elif fdr == 'bonferroni':
        alphaFDR = alpha/(nOfGenes*nDigitalSpots)
    elif fdr == 'no':
        alphaFDR = alpha

    sigGenes = []
    sigGenesWithPvals = []
    sigGenesWithTstats = []
    nSampleExperimental = sum(experimentalGroup)
    nSampleControl = len(experimentalGroup) - nSampleExperimental
    for nOfGenesChecked,actGene in enumerate(geneList):
        digitalSamplesControl = np.zeros([nDigitalSpots,(nSampleControl * nNearestNeighbors)])
        digitalSamplesExperimental = np.zeros([nDigitalSpots,(nSampleExperimental * nNearestNeighbors)])
        startControl = 0
        stopControl = nNearestNeighbors
        startExperimental = 0
        stopExperimental = nNearestNeighbors
        nTestedSamples = 0
        nControls = 0
        nExperimentals = 0
        for actSample in range(len(experiment)):
            try:
                geneIndex = experiment[actSample]['geneListMasked'].index(actGene)
            except(ValueError):
                print(f'{actGene} not in dataset')
                continue

            geneCount = np.zeros([nDigitalSpots,nNearestNeighbors])
            for spots in enumerate(experiment[actSample]['digitalSpotNearestNeighbors']):
                if np.any(spots[1] < 0):
                    geneCount[spots[0]] = np.nan
                else:
                    spotij = np.zeros([nNearestNeighbors,2], dtype=int)
                    spotij[:,1] = np.asarray(spots[1], dtype=int)
                    spotij[:,0] = geneIndex
                    
                    geneCount[spots[0]] = experiment[actSample]['filteredFeatureMatrixMasked'][spotij[:,0],spotij[:,1]]
                    
            spotCount = np.nanmean(geneCount, axis=1)
            nTestedSamples += 1
            if experimentalGroup[actSample] == 0:
                digitalSamplesControl[:,startControl:stopControl] = geneCount
                startControl += nNearestNeighbors
                stopControl += nNearestNeighbors
                nControls += 1
            elif experimentalGroup[actSample] == 1:
                digitalSamplesExperimental[:,startExperimental:stopExperimental] = geneCount
                startExperimental += nNearestNeighbors
                stopExperimental += nNearestNeighbors
                nExperimentals += 1
                
            else:
                continue
                
        digitalSamplesControl = np.array(digitalSamplesControl, dtype=float).squeeze()
        digitalSamplesExperimental = np.array(digitalSamplesExperimental, dtype=float).squeeze()
        #####################################################################################################
        # this will check that at least a certain number of spots show expression for the gene of interest  #
        # might be able to remove entirely now that FDR has been expanded                                   #
        #####################################################################################################
        checkControlSamples = np.count_nonzero(digitalSamplesControl,axis=1)
        checkExperimentalSamples = np.count_nonzero(digitalSamplesExperimental,axis=1)
        checkAllSamples = checkControlSamples & checkExperimentalSamples > 20
        if sum(checkAllSamples) < 20:
            continue
        else:
            actTtest = scipy.stats.ttest_ind(digitalSamplesExperimental,digitalSamplesControl, axis=1, nan_policy='propagate')
            actTstats = actTtest[0]
            actPvals = actTtest[1]
            mulCompResults = actPvals < alphaFDR
            if sum(mulCompResults) > 0:
                actSigGene = [actGene,sum(mulCompResults)]
                sigGenes.append(actSigGene)
                actSigGeneWithPvals = np.append(actSigGene, actPvals)
                actSigGeneWithTstats = np.append(actSigGene, actTstats)
                sigGenesWithPvals.append(actSigGeneWithPvals)
                sigGenesWithTstats.append(actSigGeneWithTstats)
                print(actGene)
            else:
                continue
    timestr = time.strftime("%Y%m%d-%H%M%S")
    with open(os.path.join(derivatives,f'listOfDEGsPvalues{fdr}Correction_{timestr}.csv'), 'w', encoding='UTF8') as f:
        writer = csv.writer(f)
        for i in sigGenesWithPvals:
            writer.writerow(i)
            
    with open(os.path.join(derivatives,f'listOfDEGsTstatistics{fdr}Correction_{timestr}.csv'), 'w', encoding='UTF8') as f:
        writer = csv.writer(f)
        for i in sigGenesWithTstats:
            writer.writerow(i)

#%% add analysis that utilizes spots of known gene
# this version uses processed but unregistered samples
def selectSpotsWithGene(processedSample, geneToSelect):
    denseMatrix = processedSample['filteredFeatureMatrixLog2']
    denseMatrix = denseMatrix.todense()
    geneIndex = processedSample['geneListMasked'].index(geneToSelect)
    actSpots = processedSample['filteredFeatureMatrixLog2'][geneIndex, :]
    actSpots = actSpots.todense()
    posSpots = actSpots > 0
    if np.sum(actSpots) > 0:
        posSpots = np.squeeze(np.array(posSpots))
        maskedTissuePositionList = processedSample['processedTissuePositionList'][posSpots,:]
        maskedMatrix = denseMatrix[:,posSpots]
    else:
        print(f"No spots in {processedSample[sampleID]} are positive for {geneToSelect}")
    return maskedMatrix






            