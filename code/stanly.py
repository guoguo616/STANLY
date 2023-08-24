#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""".
STANLY is a code base built in order to process and analyze various types of 
spatial transcriptomic data. This includes masking using images, spots, cells, etc.
This project has been created and maintained by Zeru Peterson at the University of Iowa

@author: Zeru Peterson, zeru-peterson@uiowa.edu https://research-git.uiowa.edu/zjpeters/STANLY
"""
__version__ = '0.0.1'
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
import time
import itk
import pandas as pd
from matplotlib.widgets import LassoSelector
from matplotlib.path import Path
# next few lines first grabs location of main script and uses that to get the location of the reference data, i.e. one back from teh code folder
codePath = os.path.realpath(os.path.dirname(__file__))
refDataPath = codePath.split('/')
del refDataPath[-1]
refDataPath = os.path.join('/',*refDataPath)

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
    1. import spatial transcriptomic/genomic data
    2. import relevant atlas images and annotations from allen atlas
    3. prepare data for registration into Common Coordinate Framework (CCF)
    4. use SyN registration from ANTs to register histological image to allen image
    5. bring remaining data, such as spot coordinates, into allen space using above transformations
    6. measure for nearest neighbor similarity among spots in new space and create a vector that represents the nearest neighbors from each slice
"""

def importVisiumData(sampleFolder):
    # this currently assumes that sampleFolder contains spatial folder and the
    # filtered_feature_bc_matrix.h5 output from space ranger, but includes some attempts to resolve
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
    # it would probably make more sense to go back and create the whole brain and split it into two, rather than recreating all three
    annotation_data = ants.image_read(os.path.join(refDataPath,'data','ccf','annotation_10.nrrd'))
    templateData = {}
    bestSlice = sliceLocation * 10
    templateSlice = ara_data.slice_image(0,(bestSlice))
    templateAnnotationSlice = annotation_data.slice_image(0,(bestSlice))
    templateLeft = templateSlice[:,:570]
    templateRight = templateSlice[:,570:]

    templateLeft = cv2.normalize(templateLeft, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    templateRight = cv2.normalize(templateRight, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    templateAnnotationLeft = templateAnnotationSlice[:,:570]
    templateAnnotationRight = templateAnnotationSlice[:,570:]
    templateSlice = cv2.normalize(templateSlice.numpy(), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    
    templateData['templateWholeGauss'] = filters.gaussian(templateSlice, sigma=10)
    templateData['templateLeftGauss'] = filters.gaussian(templateLeft, sigma=10)
    templateData['templateRightGauss'] = filters.gaussian(templateRight, sigma=10)
    # templateLeftSliceGauss = filters.gaussian(templateLeftSlice, 10)
    # templateRightSliceGauss = filters.gaussian(templateRightSlice, 10)
    templateData['sliceNumber'] = sliceLocation
    templateData['leftHem'] = templateLeft
    templateData['rightHem'] = templateRight
    templateData['wholeBrain'] = templateSlice
    templateData['leftHemAnnot'] = np.array(templateAnnotationLeft, dtype='int32')
    templateData['rightHemAnnot'] = np.array(templateAnnotationRight, dtype='int32')
    templateData['wholeBrainAnnot'] = np.array(templateAnnotationSlice.numpy(), dtype='int32')
    annotX = templateData['wholeBrainAnnot'].shape[0]
    annotY = templateData['wholeBrainAnnot'].shape[1]
    templateAnnotRGB = np.zeros([annotX, annotY, 3])
    templateAnnotUnique = np.unique(templateData['wholeBrainAnnot'])
    templateAnnotRenum = np.zeros(templateData['wholeBrainAnnot'].shape)
    for newNum, origNum in enumerate(templateAnnotUnique):
        templateAnnotRenum[templateData['wholeBrainAnnot'] == origNum] = newNum

    se = morphology.disk(2)
    templateAnnotRGB = morphology.binary_dilation(feature.canny(templateAnnotRenum), footprint=se)
    templateData['wholeBrainAnnotEdges']  = templateAnnotRGB
    templateData['leftHemAnnotEdges'] = templateAnnotRGB[:,:570]
    templateData['rightHemAnnotEdges'] = templateAnnotRGB[:,570:]
    # currently using the 10um resolution atlas, would need to change if that changes
    templateData['startingResolution'] = 0.01
    annotation_id = []
    annotation_name = []
    structure_id_path = []
    color_hex = []
    with open(os.path.join(codePath,'data','allen_ccf_annotation.csv'), newline='') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',')
        next(csvreader)
        for row in csvreader:
            # imports the annotation id, and full name
            annotation_id.append(row[3])
            annotation_name.append(row[4])
            structure_id_path.append(row[7])
            color_hex.append(row[1])

    templateData['annotationID'] = annotation_id
    templateData['annotationName'] = annotation_name
    templateData['structureIDPath'] = structure_id_path
    templateData['colorHex'] = color_hex
    return templateData

# tissue coordinates should reference output of importVisiumData
# rotation currently accepts 0,90,180,270, will take input from processedVisium
def rotateTissuePoints(tissuePoints, tissueImage, theta, flip=False):
    ## need to add merfish compatibility, which shouldn't be hard, just adjusting tissue position list info
    # scales tissue coordinates down to image resolution
    
    # below rotates coordinates and accounts for shift resulting from matrix rotation above, will be different for different angles
    # since the rotation is happening in euclidean space, we have to bring the coordinates back to image space
    rad = np.deg2rad(theta)
    rotMat = np.array([[np.cos(rad),-(np.sin(rad))],\
              [np.sin(rad),np.cos(rad)]])
    origin = np.array([tissueImage.shape[1]/2,tissueImage.shape[0]/2])
    rotImage = rotate(tissueImage, theta, resize=True)
    if flip==True:
        rotImage = rotImage[:,::-1]
        rotMat = np.array([[-np.cos(rad),(np.sin(rad))],\
                  [np.sin(rad),np.cos(rad)]])
    rotOrigin = np.array([rotImage.shape[1]/2, rotImage.shape[0]/2])
    centeredTP = tissuePoints - origin
    tissuePointsRotate = np.matmul(centeredTP, rotMat)
    tissuePointsRotateCenter = tissuePointsRotate + rotOrigin
   
    # plt.imshow(rotImage, cmap='gray')
    # plt.scatter(tissuePointsRotateCenter[:,0],tissuePointsRotateCenter[:,1], alpha=0.3)
    # plt.show()
    return tissuePointsRotateCenter, rotImage

# prepares visium data for registration
def processVisiumData(visiumData, templateData, rotation, outputFolder, log2normalize=True, flip=False):
    processedVisium = {}
    # the sampleID might have issues on non unix given the slash direction, might need to fix
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
    # processedVisium['tissueSpotBarcodeListSorted'] = tissueSpotBarcodeListSorted
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


def runANTsToAllenRegistration(processedData, templateData, log2normalize=True, hemisphere='wholeBrain'):
    # registeredData will contain: sampleID, derivativesPath, transformedTissuePositionList, fwdtransforms, invtransforms
    registeredData = {}
    # if hemisphere == 'rightHem':
    templateAntsImage = ants.from_numpy(templateData[hemisphere])
    maxWidth = templateData[hemisphere].shape[1]
    # elif hemisphere == 'leftHem':
    #     templateAntsImage = ants.from_numpy(templateData['leftHem'])
    #     maxWidth = templateData['lefttHem'].shape[1]
    # elif 'wholeBrain':
    #     templateAntsImage = ants.from_numpy(templateData['wholeBrain'])
    #     maxWidth = templateData['wholeBrain'].shape[1]
    try:
# <<<<<<< HEAD
#         file = open(f"{os.path.join(processedVisium['derivativesPath'],processedVisium['sampleID'])}_tissuePointsProcessedToAllen.csv", 'r')
#         print(f"{processedVisium['sampleID']} has already been processed and is located at: {processedVisium['derivativesPath']}")
#         print(f"Loading data for {processedVisium['sampleID']}")
#         registeredData['sampleID'] = processedVisium['sampleID']
#         registeredData['derivativesPath'] = processedVisium['derivativesPath']
#         registeredData['geneListMasked'] = processedVisium['geneListMasked']
#         registeredData['fwdtransforms'] = [os.path.join(processedVisium['derivativesPath'],f"{processedVisium['sampleID']}_xfm1Warp.nii.gz"),os.path.join(processedVisium['derivativesPath'],f"{processedVisium['sampleID']}_xfm0GenericAffine.mat")]
#         registeredData['invtransforms'] = [os.path.join(processedVisium['derivativesPath'],f"{processedVisium['sampleID']}_xfm0GenericAffine.mat"), os.path.join(processedVisium['derivativesPath'],f"{processedVisium['sampleID']}_xfm1InverseWarp.nii.gz"),]
#         registeredData['tissueRegistered'] = plt.imread(os.path.join(processedVisium['derivativesPath'],f"{processedVisium['sampleID']}_tissue_registered_to_Allen_slice_{templateData['sliceNumber']}.png"))
#     except Exception as e:
#         print(f"Registering {processedVisium['sampleID']}")
# =======
        file = open(f"{os.path.join(processedData['derivativesPath'],processedData['sampleID'])}_tissuePointsProcessedToAllen.csv", 'r')
        print(f"{processedData['sampleID']} has already been processed and is located at: {processedData['derivativesPath']}")
        print(f"Loading data for {processedData['sampleID']}")
        registeredData['sampleID'] = processedData['sampleID']
        registeredData['derivativesPath'] = processedData['derivativesPath']
        registeredData['geneListMasked'] = processedData['geneListMasked']
        registeredData['fwdtransforms'] = [os.path.join(processedData['derivativesPath'],f"{processedData['sampleID']}_xfm1Warp.nii.gz"),os.path.join(processedData['derivativesPath'],f"{processedData['sampleID']}_xfm0GenericAffine.mat")]
        registeredData['invtransforms'] = [os.path.join(processedData['derivativesPath'],f"{processedData['sampleID']}_xfm0GenericAffine.mat"), os.path.join(processedData['derivativesPath'],f"{processedData['sampleID']}_xfm1InverseWarp.nii.gz"),]
        registeredData['tissueRegistered'] = plt.imread(os.path.join(processedData['derivativesPath'],f"{processedData['sampleID']}_tissue_registered_to_Allen_slice_{templateData['sliceNumber']}.png"))
    except Exception as e:
        #######################################################################
        # working here to add merfish updates
        #######################################################################
        print(f"Registering {processedData['sampleID']}")
# >>>>>>> merfishTesting
        # convert into ants image type
        registeredData['sampleID'] = processedData['sampleID']
        registeredData['derivativesPath'] = processedData['derivativesPath']
        registeredData['geneListMasked'] = processedData['geneListMasked']
        
        sampleAntsImage = ants.from_numpy(processedData['tissueProcessed'])
        # run registration
        synXfm = ants.registration(fixed=templateAntsImage, moving=sampleAntsImage, \
        type_of_transform='SyNAggro', grad_step=0.1, reg_iterations=(120, 100,80,60,40,20,0), \
        syn_sampling=32, flow_sigma=3,syn_metric='mattes', outprefix=os.path.join(processedData['derivativesPath'],f"{processedData['sampleID']}_xfm"))
        
        #registeredData['antsOutput'] = synXfm
        registeredData['fwdtransforms'] = synXfm['fwdtransforms']
        registeredData['invtransforms'] = synXfm['invtransforms']

        # apply syn transform to tissue spot coordinates
        applyTransformStr = f"antsApplyTransformsToPoints -d 2 -i {os.path.join(processedData['derivativesPath'],processedData['sampleID'])}_tissuePointsProcessed.csv -o {os.path.join(processedData['derivativesPath'],processedData['sampleID'])}_tissuePointsProcessedToAllen.csv -t [ {os.path.join(processedData['derivativesPath'],processedData['sampleID'])}_xfm0GenericAffine.mat,1] -t [{os.path.join(processedData['derivativesPath'],processedData['sampleID'])}_xfm1InverseWarp.nii.gz]"
        pid = os.system(applyTransformStr)
        # program has to wait while spots are transformed by the system
        if pid:
            os.wait()
        registeredData['tissueRegistered'] = synXfm["warpedmovout"].numpy()

    transformedTissuePositionList = []
    with open(os.path.join(f"{os.path.join(processedData['derivativesPath'],processedData['sampleID'])}_tissuePointsProcessedToAllen.csv"), newline='') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',')
            next(csvreader)
            for row in csvreader:
                transformedTissuePositionList.append(row)
                

    transformedTissuePositionList = np.array(transformedTissuePositionList, dtype='float32')
    # switching x,y columns back to python compatible and deleting empty columns
    transformedTissuePositionList[:,[0,1]] = transformedTissuePositionList[:,[1,0]]
    registeredData['transformedTissuePositionList'] = np.delete(transformedTissuePositionList, [2,3,4,5],1)

    # plt.imshow(registeredData['tissueRegistered'])
    # plt.scatter(registeredData['transformedTissuePositionList'][0:,0],registeredData['transformedTissuePositionList'][0:,1], marker='.', c='red', alpha=0.3)
    # plt.show()
    
    # plt.imshow(registeredData['tissueRegistered'],cmap='gray')
    # plt.imshow(templateData['rightHem'], alpha=0.3)
    # plt.title(processedVisium['sampleID'])
    # plt.show()
        
    transformedTissuePositionListMask = np.logical_and(registeredData['transformedTissuePositionList'] > 0, registeredData['transformedTissuePositionList'] < maxWidth)
    maskedTissuePositionList = []
    geneMatrixMaskedIdx = []
    for i, masked in enumerate(transformedTissuePositionListMask):
        if masked.all() == True:
            geneMatrixMaskedIdx.append(i)
            maskedTissuePositionList.append(registeredData['transformedTissuePositionList'][i])

    registeredData['maskedTissuePositionList'] = np.array(maskedTissuePositionList, dtype='float32')

    # registeredData['filteredFeatureMatrixMasked'] = np.delete(filteredFeatureMatrixMasked, 0,1)
    if log2normalize == True:
        tempDenseMatrix = processedData['geneMatrixLog2'].todense().astype('float32')
    else:
        tempDenseMatrix = processedData['geneMatrix'].todense().astype('float32')
    registeredData['geneMatrixMasked'] = sp_sparse.csc_matrix(tempDenseMatrix[:,geneMatrixMaskedIdx])
    # write re-ordered filtered feature matrix csv to match tissue spot order
    sp_sparse.save_npz(f"{os.path.join(processedData['derivativesPath'],processedData['sampleID'])}_OrderedLog2FeatureMatrixTemplateMasked.npz", sp_sparse.csc_matrix(registeredData['geneMatrixMasked']))        
    cv2.imwrite(f"{registeredData['derivativesPath']}/{registeredData['sampleID']}_tissue_registered_to_Allen_slice_{templateData['sliceNumber']}.png",registeredData['tissueRegistered'])
    
    return registeredData

def runANTsInterSampleRegistration(processedVisium, sampleToRegisterTo, log2normalize=True):
    # convert into ants image type
    registeredData = {}
    templateAntsImage = ants.from_numpy(sampleToRegisterTo['tissueProcessed'])
    sampleAntsImage = ants.from_numpy(processedVisium['tissueProcessed'])
    # mattes seems to be most conservative syn_metric
    synXfm = ants.registration(fixed=templateAntsImage, moving=sampleAntsImage, \
    type_of_transform='SyNAggro', grad_step=0.1, reg_iterations=(120, 100,80,60,40,20,0), \
    syn_sampling=32, flow_sigma=3, syn_metric='mattes', outprefix=os.path.join(processedVisium['derivativesPath'],f"{processedVisium['sampleID']}_to_{sampleToRegisterTo['sampleID']}_xfm"))
    # registeredData['antsOutput'] = synXfm
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
                
    registeredData['tissueRegistered'] = synXfm["warpedmovout"].numpy()
    # registeredData['filteredFeatureMatrixGeneList'] = processedVisium['filteredFeatureMatrixGeneList']
    registeredData['geneListMasked'] = processedVisium['geneListMasked']

    registeredData['transformedTissuePositionList'] = np.array(transformedTissuePositionList, dtype='float32')
    # switching x,y columns back to python compatible and deleting empty columns
    registeredData['transformedTissuePositionList'][:,[0,1]] = registeredData['transformedTissuePositionList'][:,[1,0]]
    registeredData['transformedTissuePositionList'] = np.delete(registeredData['transformedTissuePositionList'], [2,3,4,5],1)
    # registeredData['tissueSpotBarcodeList'] = processedVisium['tissueSpotBarcodeList']
    if log2normalize==True:
        registeredData['geneMatrixLog2'] = processedVisium['geneMatrixLog2']
    else:
        registeredData['geneMatrix'] = processedVisium['geneMatrix']
    plt.imshow(registeredData['tissueRegistered'], cmap='gray')
    plt.scatter(registeredData['transformedTissuePositionList'][0:,0],registeredData['transformedTissuePositionList'][0:,1], marker='.', c='red', alpha=0.3)
    plt.show()
    
    plt.imshow(sampleToRegisterTo['tissueProcessed'], cmap='gray')
    plt.imshow(registeredData['tissueRegistered'], alpha=0.7, cmap='gray')
    plt.title(processedVisium['sampleID'])
    plt.show()

    cv2.imwrite(f"{registeredData['derivativesPath']}/{registeredData['sampleID']}_registered_to_{sampleToRegisterTo['sampleID']}.png",registeredData['tissueRegistered'])

    return registeredData
    
def applyAntsTransformations(registeredVisium, bestSampleRegisteredToTemplate, templateData, log2normalize=True, hemisphere='wholeBrain'):
    # if not os.exists(f"{os.path.join(registeredVisium['derivativesPath'],registeredVisium['sampleID'])}_tissuePointOrderedFeatureMatrixTemplateMasked.csv"):
        
    templateAntsImage = ants.from_numpy(templateData[hemisphere])
    sampleAntsImage = ants.from_numpy(registeredVisium['tissueRegistered'])
    sampleToTemplate = ants.apply_transforms( fixed=templateAntsImage, moving=sampleAntsImage, transformlist=bestSampleRegisteredToTemplate['fwdtransforms'])
    
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
    templateRegisteredData['tissueRegistered'] = sampleToTemplate.numpy()

    transformedTissuePositionList = np.array(transformedTissuePositionList, dtype='float32')
    # switching x,y columns back to python compatible and deleting empty columns
    transformedTissuePositionList[:,[0,1]] = transformedTissuePositionList[:,[1,0]]
    transformedTissuePositionList = np.delete(transformedTissuePositionList, [2,3,4,5],1)
    templateRegisteredData['geneListMasked'] = registeredVisium['geneListMasked']

    plt.imshow(templateRegisteredData['tissueRegistered'], cmap='gray')
    plt.scatter(transformedTissuePositionList[0:,0],transformedTissuePositionList[0:,1], marker='.', c='red', alpha=0.3)
    plt.show()

    plt.imshow(templateData['rightHem'], cmap='gray')    
    plt.imshow(templateRegisteredData['tissueRegistered'],alpha=0.8,cmap='gray')
    plt.title(templateRegisteredData['sampleID'])
    plt.show()
        
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
        tempDenseMatrix = registeredVisium['geneMatrixLog2'].todense().astype('float32')
    else:
        tempDenseMatrix = registeredVisium['geneMatrix'].todense().astype('float32')
    templateRegisteredData['geneMatrixMasked'] = sp_sparse.csr_matrix(tempDenseMatrix[:,geneMatrixMaskedIdx])
    # imageFilename = f"os.path.join({registeredVisium['derivativesPath']},{registeredVisium['sampleID'])}_registered_to_{bestSampleRegisteredToTemplate['sampleID']}_to_Allen.png"
    # os.path.join(processedData['derivativesPath'],f"{processedData['sampleID']}_tissue_registered_to_Allen_slice_{templateData['sliceNumber']}.png"
    imageFilename = os.path.join(registeredVisium['derivativesPath'],f"{registeredVisium['sampleID']}_registered_to_{bestSampleRegisteredToTemplate['sampleID']}_to_Allen_slice_{templateData['sliceNumber']}.png")
    cv2.imwrite(imageFilename, templateRegisteredData['tissueRegistered'])

    return templateRegisteredData

# create digital spots for an allen template slice, with spot size defined in 10um
def createDigitalSpots(templateRegisteredData, desiredSpotSize):
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

    # remove non-tissue spots
    roundedTemplateSpots = np.array(templateSpots.round(), dtype='int32')
    ### the following line is dependent on bestSampleToTemplate, so either fix dependency or make input be bestSampleToTemplate
    digitalSpots = []
    for row in range(len(roundedTemplateSpots)):
        if templateRegisteredData['tissueRegistered'][roundedTemplateSpots[row,1],roundedTemplateSpots[row,0]] > 0:
            digitalSpots.append(templateSpots[row])
            
    digitalSpots = np.array(digitalSpots)
    # uncomment following 3 lines to see the digital template spots
    plt.imshow(templateRegisteredData['tissueRegistered'])
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
    z = 0
    for actSpot in digitalSpots:
        spotCdist = sp_spatial.distance.cdist(templateRegisteredSpots, np.array(actSpot).reshape(1,-1), 'euclidean')
        sortedSpotCdist = np.sort(spotCdist, axis=0)
        actSpotCdist = sortedSpotCdist[0:kNN]
        # spotNNIdx gives the index of the top kSpots nearest neighbors for each digital spot
        spotMeanCdist = np.mean(actSpotCdist)
        blankIdx = np.zeros([kNN,1], dtype='int32')
        blankIdx[:] = -9999
        spotNNIdx = []
        print(z)
        for i in actSpotCdist:
            if spotMeanCdist < (spotDist * 3) and np.all(i):
                actNNIdx = np.array(np.where(spotCdist == i)[0],dtype='int32')
                spotNNIdx.append(actNNIdx[:])
            else:
                # should probably change this from 0s to something like -1
                spotNNIdx = blankIdx
            
        allMeanCdists.append(spotMeanCdist)
        allSpotNN.append(np.array(spotNNIdx))
        z = z + 1
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
    regionMask = np.zeros(template['rightHemAnnot'].shape, dtype='int')  
    # maskList = []
    for actID in structIDs:
        # structMask = np.zeros(template['rightHemAnnot'].shape, dtype='int')        
        regionBoolMask = np.where(template['rightHemAnnot'] == int(actID))
        if any(regionBoolMask[0]):
            regionMask[regionBoolMask[0],regionBoolMask[1]] = 1
        # regionMask = regionMask + structMask
        
    plt.imshow(template['rightHem'], cmap='gray')
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
    roundedTemplateSpots = np.array(templateSpots.round(), dtype='int32')
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

def loadProcessedVisiumSample(locOfProcessedSample, loadLog2Norm=True):
    processedVisium = {}
    processedVisium['derivativesPath'] = os.path.join(locOfProcessedSample)
    processedVisium['sampleID'] = locOfProcessedSample.rsplit(sep='/',maxsplit=1)[-1]
    processedVisium['sourceType'] = 'visium'
    processedVisium['tissueProcessed'] = io.imread(os.path.join(processedVisium['derivativesPath'],f"{processedVisium['sampleID']}_tissueProcessed.png"))
    jsonPath = open(os.path.join(processedVisium['derivativesPath'],f"{processedVisium['sampleID']}_processing_information.json"))
    processedSampleJson = json.loads(jsonPath.read())
    processedVisium['geneListMasked'] = processedSampleJson['geneList']
    processedVisium['spotCount'] = processedSampleJson['spotCount']
    if loadLog2Norm==True:
        geneMatrixLog2 = sp_sparse.load_npz(os.path.join(processedVisium['derivativesPath'], f"{processedVisium['sampleID']}_tissuePointOrderedFeatureMatrixLog2Normalized.npz"))
        processedVisium['geneMatrixLog2'] = geneMatrixLog2
    else:
        geneMatrix = sp_sparse.load_npz(os.path.join(processedVisium['derivativesPath'], f"{processedVisium['sampleID']}_tissuePointOrderedFeatureMatrix.npz"))
        processedVisium['geneMatrix'] = geneMatrix
    tissuePositionList = []
    with open(os.path.join(f"{os.path.join(processedVisium['derivativesPath'],processedVisium['sampleID'])}_tissuePointsProcessed.csv"), newline='') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',')
            next(csvreader)
            for row in csvreader:
                tissuePositionList.append(row)
                
    tissuePositionList = np.array(tissuePositionList, dtype='float32')
    # switching x,y columns back to python compatible and deleting empty columns
    tissuePositionList[:,[0,1]] = tissuePositionList[:,[1,0]]
    processedVisium['processedTissuePositionList'] = np.delete(tissuePositionList, [2,3,4,5],1)
    return processedVisium

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
    with open(os.path.join(f"{os.path.join(templateRegisteredData['derivativesPath'],templateRegisteredData['sampleID'])}_tissuePointsResize_to_{templateRegisteredData['bestFitSampleID']}TemplateTransformApplied.csv"), newline='') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',')
            next(csvreader)
            for row in csvreader:
                transformedTissuePositionList.append(row)
    transformedTissuePositionList = np.array(transformedTissuePositionList, dtype='float32')
    transformedTissuePositionList[:,[0,1]] = transformedTissuePositionList[:,[1,0]]
    transformedTissuePositionList = np.delete(transformedTissuePositionList, [2,3,4,5],1)
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

# these could probably be included as part of a class for both processed and registered samples

def viewGeneInProcessedVisium(processedSample, geneName):
    try:
        geneIndex = processedSample['geneListMasked'].index(geneName)
        actSpots = processedSample['geneMatrixLog2'][geneIndex, :]
        plt.imshow(processedSample['tissueProcessed'], cmap='gray')
        plt.scatter(processedSample['processedTissuePositionList'][:,0],processedSample['processedTissuePositionList'][:,1], c=np.array(actSpots.todense()), alpha=0.8, cmap='Reds', marker='.')
        plt.title(f'Gene count for {geneName} in {processedSample["sampleID"]}')
        plt.colorbar()
        # plt.savefig(os.path.join(derivatives,f'geneCount{geneName}{processedSample["sampleID"]}Registered.png'), bbox_inches='tight', dpi=300)
        plt.show()
    except(ValueError):
        print(f'{geneName} not found in dataset')
def viewGeneInRegisteredVisium(registeredSample, geneName):
    try:
        geneIndex = registeredSample['geneListMasked'].index(geneName)
        actSpots = registeredSample['geneMatrixLog2'][geneIndex, :]
        plt.imshow(registeredSample['tissueProcessed'], cmap='gray')
        plt.scatter(registeredSample['processedTissuePositionList'][:,0],registeredSample['processedTissuePositionList'][:,1], c=np.array(actSpots.todense()), alpha=0.8, cmap='Reds', marker='.')
        plt.title(f'Gene count for {geneName} in {processedSample["sampleID"]}')
        plt.colorbar()
        # plt.savefig(os.path.join(derivatives,f'geneCount{geneName}{processedSample["sampleID"]}Registered.png'), bbox_inches='tight', dpi=300)
        plt.show()
    except(ValueError):
        print(f'{geneName} not found in dataset')

#%% not working properly indpendently yet
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
                    spotij = np.zeros([nNearestNeighbors,2], dtype='int32')
                    spotij[:,1] = np.asarray(spots[1], dtype='int32')
                    spotij[:,0] = geneIndex
                    
                    geneCount[spots[0]] = experiment[actSample]['geneMatrixMasked'][spotij[:,0],spotij[:,1]]
                    
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
                
        digitalSamplesControl = np.array(digitalSamplesControl, dtype='float32').squeeze()
        digitalSamplesExperimental = np.array(digitalSamplesExperimental, dtype='float32').squeeze()
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
        print(f"No spots in {processedSample[sampleID]} are positive for {geneToSelect}")
    return maskedMatrix, maskedTissuePositionList
           
#%% calculate the cosine similarity of a given matrix at the coordinates given
def cosineSimOfConnection(inputMatrix,i,j):
    I = inputMatrix[:,i]
    J = inputMatrix[:,j]
    # cs = np.sum(np.dot(I,J.transpose())) / (np.sqrt(np.sum(np.square(I)))*np.sqrt(np.sum(np.square(J))))
    cs = sp_spatial.distance.cosine(I,J)
    return cs

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
    
def importMerfishData(sampleFolder, outputPath):
    # 
    sampleData = {}
    if os.path.exists(os.path.join(sampleFolder)):
        # spatialFolder = os.path.join(sampleFolder)
        if any(glob(os.path.join(sampleFolder, '*cell_by_gene*.csv'))):
            # need to check how the cell by gene file is usually output/named
            # os.path.isfile(glob(os.path.join(sampleFolder, '*cell_by_gene_*.csv'))[0])
            dataFolder = sampleFolder
            sampleData['sampleID'] = sampleFolder.rsplit(sep='/',maxsplit=1)[-1]
        elif any(glob(os.path.join(sampleFolder, 'region_0','*cell_by_gene*.csv'))):
            
            os.path.isfile(glob(os.path.join(sampleFolder,'region_0','*cell_by_gene*.csv'))[0])
            dataFolder = os.path.join(sampleFolder, 'region_0')
            sampleData['sampleID'] = sampleFolder.rsplit(sep='/',maxsplit=1)[-1]
            # os.path.isfile(glob(os.path.join(spatialFolder, '*filtered_feature_bc_matrix.h5'))[0])
            # dataFolder = spatialFolder
        else:
            print("Something is wrong!")
        # dataFolder = os.path.join(sampleFolder)
    else:
        print(f"{sampleFolder} not found!")
    # need to look for standard outputname of tiff file for registration
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
    # check if image has already been downsampled
    if any(glob(downsampledImagePath)):
        print(f"Loading previously downsampled image from {dataFolder}")
    else:
        print("Downsampling high resolution image to 10 micron resolution")
        downsampleMerfishTiff(originalImagePath, downsampledImagePath, scale=0.01)
    sampleData['imageData'] = io.imread(downsampledImagePath)
    # need to convert into 0-1 
    # sampleImageNorm = (sampleData['imageData'] - np.min(sampleData['imageData']))/(np.max(sampleData['imageData']) - np.min(sampleData['imageData']))
    sampleData['imageDataGray'] = np.array((sampleData['imageData'] - np.min(sampleData['imageData']))/(np.max(sampleData['imageData']) - np.min(sampleData['imageData'])), dtype='float32')
    
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
    ### no scale factor equivalent that I know of in merfish, but using nanometer as reference can approximate scaling so far
    # scaleFactorPath = open(os.path.join(spatialFolder,"scalefactors_json.json"))
    # sampleData['scaleFactors'] = json.loads(scaleFactorPath.read())
    # scaleFactorPath.close()
    geneMatrixPath = glob(os.path.join(dataFolder,"*cell_by_gene*.csv"))[0]
    geneMatrix = []
    with open(geneMatrixPath, newline='') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',')
        next(csvreader)
        for row in csvreader:
            geneMatrix.append(np.array(row[1:], dtype='float32'))
    sampleData['geneMatrix'] = np.array(geneMatrix, dtype='int32')
    # the ratio of real spot diameter, 55um, by imaged resolution of spot
    # sampleData['spotStartingResolution'] = 0.55 / visiumData["scaleFactors"]["spot_diameter_fullres"]
    cellByGene = pd.read_csv(geneMatrixPath)
    geneList = cellByGene.columns[1:]
    sampleData['geneList'] = list(geneList) 
    # plt.imshow(visiumData['imageData'])
    return sampleData

def loadProcessedMerfishSample(locOfProcessedSample, loadLog2Norm=True):
    processedSample = {}
    processedSample['derivativesPath'] = os.path.join(locOfProcessedSample)
    processedSample['sampleID'] = locOfProcessedSample.rsplit(sep='/',maxsplit=1)[-1]
    processedSample['sourceType'] = 'merfish'
    processedSample['tissueProcessed'] = io.imread(os.path.join(processedSample['derivativesPath'],f"{processedSample['sampleID']}_tissueProcessed.png"))
    jsonPath = open(os.path.join(processedSample['derivativesPath'],f"{processedSample['sampleID']}_processing_information.json"))
    processedSampleJson = json.loads(jsonPath.read())
    processedSample['geneListMasked'] = processedSampleJson['geneList']
    # processedSample['spotCount'] = processedSampleJson['spotCount']
    if loadLog2Norm==True:
        geneMatrixLog2 = sp_sparse.load_npz(os.path.join(processedSample['derivativesPath'], f"{processedSample['sampleID']}_tissuePointOrderedFeatureMatrixLog2Normalized.npz"))
        processedSample['geneMatrixLog2'] = geneMatrixLog2
    else:
        geneMatrix = sp_sparse.load_npz(os.path.join(processedSample['derivativesPath'], f"{processedSample['sampleID']}_tissuePointOrderedFeatureMatrix.npz"))
        processedSample['geneMatrix'] = geneMatrix
    tissuePositionList = []
    with open(os.path.join(f"{os.path.join(processedSample['derivativesPath'],processedSample['sampleID'])}_tissuePointsProcessed.csv"), newline='') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',')
            next(csvreader)
            for row in csvreader:
                tissuePositionList.append(row)
                
    tissuePositionList = np.array(tissuePositionList, dtype='float32')
    # switching x,y columns back to python compatible and deleting empty columns
    tissuePositionList[:,[0,1]] = tissuePositionList[:,[1,0]]
    processedSample['processedTissuePositionList'] = np.delete(tissuePositionList, [2,3,4,5],1)
    return processedSample


def processMerfishData(sampleData, templateData, rotation, outputFolder, log2normalize=True):
    processedData = {}
    # the sampleID might have issues on non unix given the slash direction, might need to fix
    processedData['sampleID'] = sampleData['sampleID']
    processedData['sourceType'] = 'merfish'
    processedData['tissuePositionList'] = sampleData['tissuePositionList']
    processedData['geneMatrix'] = np.transpose(sampleData['geneMatrix'])
    processedData['geneList'] = sampleData['geneList']
    outputPath = os.path.join(outputFolder, sampleData['sampleID'])
    #### need to create a loadProcessedMerfishData function
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
    #### shouldn't need otsu for merfish, since the background has already been removed, need to confirm
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
    processedDataDict = {
        "sampleID": sampleData['sampleID'],
        "rotation": int(rotation),
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
    
    header=['x','y','z','t','label','comment']
    rowFormat = []
    # the x and y are swapped between ants and numpy, but this is so far dealt with within the code
    with open(f"{os.path.join(outputPath,processedData['sampleID'])}_tissuePointsProcessed.csv", 'w', encoding='UTF8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i in range(len(processedData['processedTissuePositionList'])):
            rowFormat = [processedData['processedTissuePositionList'][i,1]] + [processedData['processedTissuePositionList'][i,0]] + [0] + [0] + [0] + [0]
            writer.writerow(rowFormat)
    return processedData

#%% cell type identification

def selectSpotsWithGeneList(processedSample, geneList, threshold=1):
    # threshold decides percentage of genes in list necessary
    geneListIdx = []
    for actGene in geneList:
        try:
            geneListIdx.append(processedSample['geneListMasked'].index(actGene))
        except ValueError:
            print(f"No spots are positive for {actGene}!")
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
    return maskedMatrix, maskedTissuePositionList

#%% used example script from matplotlib page as template for below
# https://matplotlib.org/stable/gallery/widgets/lasso_selector_demo_sgskip.html
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
        # self.pts = self.pts
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
            print(f"If you want to create a new sample, either rename the mask or delete previous version")
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
            # cv2.imwrite(f"{processedSampleMasked['derivativesPath']}/{processedSampleMasked['sampleID']}_tissue.png",255*sampleData['imageDataGray'])
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
        