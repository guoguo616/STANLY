#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 14:37:22 2023

@author: zjpeters
"""
import os
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from skimage.transform import rescale, rotate, resize
import itk
import sys
sys.path.insert(0, "/home/zjpeters/rdss_tnj/visiumalignment/code")
import stanly
from glob import glob
from skimage import io, filters, color, feature, morphology
import csv
import cv2
from skimage.exposure import match_histograms
import scipy.sparse as sp_sparse
import json
rawdata, derivatives = stanly.setExperimentalFolder("/home/zjpeters/rdss_tnj/visiumalignment")
sourcedata = os.path.join('/','home','zjpeters','rdss_tnj','visiumalignment','sourcedata','merscopedata')
#%% location of merfish csv data 
# datasets_mouse_brain_map_BrainReceptorShowcase_Slice1_Replicate1_
# locOfCellByGeneCsv = glob(os.path.join(sourcedata,'*cell_by_gene*.csv'))[0]
# #datasets_mouse_brain_map_BrainReceptorShowcase_Slice1_Replicate1_
# locOfCellMetadataCsv = glob(os.path.join(sourcedata,'*cell_metadata*.csv'))[0]
# # datasets_mouse_brain_map_BrainReceptorShowcase_Slice1_Replicate1_images_
tifFilename = glob(os.path.join(sourcedata,'images','*mosaic_DAPI_z0.tif'))[0]
# # load data as pandas dataframe and extract list of genes
# cellByGene = pd.read_csv(locOfCellByGeneCsv)
# cellMetadata = pd.read_csv(locOfCellMetadataCsv)
# geneList = cellByGene.columns[1:]

#%% display all genes
# for actGene in geneList:
#     plt.scatter(cellMetadata.center_x,cellMetadata.center_y, c=cellByGene[actGene], marker='.', cmap='Reds')
#     plt.title(actGene)
#     plt.gca().invert_yaxis()
#     plt.show()

#%% new functions

# def downsampleMerfishTiff(merfishImageFilename, outputName, scale=0.01):
#     # the default scale is based on merscope claiming nanometer resolution, scaled for the ccf 10um resolution    
#     img = itk.imread(merfishImageFilename)
#     imgSize = itk.size(img)
#     imgSpacing = itk.spacing(img)
#     imgOrigin = itk.origin(img)
#     imgDimension = img.GetImageDimension()
#     outSize = [int(imgSize[d] * scale) for d in range(imgDimension)]
#     outSpacing = [imgSpacing[d] / scale for d in range(imgDimension)]
#     outOrigin = [imgOrigin[d] + 0.5 * (outSpacing[d] - imgSpacing[d])
#                       for d in range(imgDimension)]
    
#     interpolator = itk.LinearInterpolateImageFunction.New(img)
    
#     resampled = itk.resample_image_filter(
#         img,
#         interpolator=interpolator,
#         size=outSize,
#         output_spacing=outSpacing,
#         output_origin=outOrigin,
#     )
#     outFilename = os.path.join(outputName)
#     itk.imwrite(resampled, outFilename)

# starting from the importVisiumData and processVisiumData function, create merfish equivalents
# expected merfish data includes:
# 1. image data, needs downsampled due to very high resolution
# 2. cell_by_gene.csv containing cell index and barcode in first two columns followed by columns of rna expression per gene per cell
# 3. cell_metadata.csv containing cell index, barcode, fov, volume, center_x, center_y, min_x, min_y, max_x, max_y
# 

#%% using itk to load and downsample tiff

# stanly.downsampleMerfishTiff(tifFilename, os.path.join(derivatives,'downsampledMerfish.tif'), scale=0.01)
# testSampleImg = io.imread(os.path.join(derivatives,'downsampledMerfish.tif'), as_gray=True)

# need to look back over code for downsampling to see why scale of 0.01 requires multiplying by scale * 9, like below
# newX = cellMetadata.center_x * 0.09
# newY = cellMetadata.center_y * 0.09
# plt.imshow(testSampleImg)
# plt.scatter(newX,newY, c=cellByGene['Gpr22'], marker='.', cmap='Reds', alpha=0.4)
# plt.show()

# def importMerfishData(sampleFolder, outputPath):
#     # 
#     sampleData = {}
#     if os.path.exists(os.path.join(sampleFolder)):
#         # spatialFolder = os.path.join(sampleFolder)
#         try:
#             # need to check how the cell by gene file is usually output/named
#             os.path.isfile(glob(os.path.join(sampleFolder, '*cell_by_gene_*.csv'))[0])
#             dataFolder = sampleFolder
#             sampleData['sampleID'] = sampleFolder.rsplit(sep='/',maxsplit=1)[-1]
#         except IndexError:
#             print("Something is wrong!")
#             # os.path.isfile(glob(os.path.join(spatialFolder, '*filtered_feature_bc_matrix.h5'))[0])
#             # dataFolder = spatialFolder
#         # dataFolder = os.path.join(sampleFolder)
#     else:
#         print(f"{sampleFolder} not found!")
#     # need to look for standard outputname of tiff file for registration
    
#     if any(glob(os.path.join(dataFolder,"*mosaic_DAPI_z0.tif"))):
#         originalImagePath =  glob(os.path.join(dataFolder,"*mosaic_DAPI_z0.tif"))[0]
#     elif any(glob(os.path.join(dataFolder,"images","*mosaic_DAPI_z0.tif"))):
#         originalImagePath =  glob(os.path.join(dataFolder,"images","*mosaic_DAPI_z0.tif"))[0]
#     else:
#         print(f"Can't find tif in {dataFolder} or {dataFolder}/images")
#     downsampledImagePath = os.path.splitext(originalImagePath)[0]
#     downsampledImagePath = downsampledImagePath + "_downsampled.tif"
#     # check if image has already been downsampled
#     if any(glob(downsampledImagePath)):
#         print(f"Loading previously downsampled image from {dataFolder}")
#     else:
#         print("Downsampling high resolution image to 10 micron resolution")
#         stanly.downsampleMerfishTiff(originalImagePath, downsampledImagePath, scale=0.01)
#     sampleData['imageData'] = io.imread(downsampledImagePath)
#     # need to convert into 0-1 
#     # sampleImageNorm = (sampleData['imageData'] - np.min(sampleData['imageData']))/(np.max(sampleData['imageData']) - np.min(sampleData['imageData']))
#     sampleData['imageDataGray'] = np.array((sampleData['imageData'] - np.min(sampleData['imageData']))/(np.max(sampleData['imageData']) - np.min(sampleData['imageData'])), dtype='float32')
    
#     cellMetadataCsv = glob(os.path.join(dataFolder,"*cell_metadata*.csv"))[0]
#     tissuePositionsList = []
#     tissueSpotBarcodes = []    
#     with open(cellMetadataCsv, newline='') as csvfile:
#         csvreader = csv.reader(csvfile, delimiter=',')
#         next(csvreader)
#         for row in csvreader:
#             tissueSpotBarcodes.append(row[0])
#             tissuePositionsList.append(row[3:5])
#     tissuePositionsList = np.array(tissuePositionsList, dtype='float32')
#     sampleData['tissueSpotBarcodeList'] = tissueSpotBarcodes
#     sampleData['tissuePositionsList'] = tissuePositionsList
#     ### no scale factor equivalent that I know of in merfish, but using nanometer as reference can approximate scaling so far
#     # scaleFactorPath = open(os.path.join(spatialFolder,"scalefactors_json.json"))
#     # sampleData['scaleFactors'] = json.loads(scaleFactorPath.read())
#     # scaleFactorPath.close()
#     geneMatrixPath = glob(os.path.join(dataFolder,"*cell_by_gene*.csv"))[0]
#     geneMatrix = []
#     with open(geneMatrixPath, newline='') as csvfile:
#         csvreader = csv.reader(csvfile, delimiter=',')
#         next(csvreader)
#         for row in csvreader:
#             geneMatrix.append(np.array(row[1:], dtype='float32'))
#     sampleData['geneMatrix'] = np.array(geneMatrix, dtype='int32')
#     # the ratio of real spot diameter, 55um, by imaged resolution of spot
#     # sampleData['spotStartingResolution'] = 0.55 / visiumData["scaleFactors"]["spot_diameter_fullres"]
#     cellByGene = pd.read_csv(geneMatrixPath)
#     geneList = cellByGene.columns[1:]
#     sampleData['geneList'] = geneList
#     # plt.imshow(visiumData['imageData'])
#     return sampleData

# def processMerfishData(sampleData, templateData, rotation, outputFolder, log2normalize=True):
#     processedData = {}
#     # the sampleID might have issues on non unix given the slash direction, might need to fix
#     processedData['sampleID'] = sampleData['sampleID']
#     processedData['tissuePositionsList'] = sampleData['tissuePositionsList']
#     processedData['geneMatrix'] = sampleData['geneMatrix']
#     processedData['geneList'] = sampleData['geneList']
#     outputPath = os.path.join(outputFolder, sampleData['sampleID'])
#     #### need to create a loadProcessedMerfishData function
#     # try:
#     #     file = open(f"{os.path.join(outputPath,processedData['sampleID'])}_tissuePointsProcessed.csv", 'r')
#     #     print(f"{processedData['sampleID']} has already been processed! Loading data")
#     #     processedData = loadProcessedSample(outputPath)
#     #     return processedData
#     # except IOError:
#     #     print(f"Processing {processedVisium['sampleID']}")
#     if not os.path.exists(outputPath):
#         os.makedirs(outputPath)
#     ### need to update when we get better resolution information
#     # resolutionRatio = visiumData['spotStartingResolution'] / templateData['startingResolution']
#     processedData['derivativesPath'] = outputPath
#     # processedVisium['tissueSpotBarcodeList'] = visiumData['tissueSpotBarcodeList']
#     # processedVisium['degreesOfRotation'] = int(rotation)
#     # first gaussian is to calculate otsu threshold
#     sampleGauss = filters.gaussian(sampleData['imageDataGray'], sigma=20)
#     otsuThreshold = filters.threshold_otsu(sampleGauss)
#     # changed to > for inv green channel, originally < below
#     processedData['visiumOtsu'] = sampleGauss > otsuThreshold
#     tissue = np.zeros(sampleGauss.shape, dtype='float32')
#     tissue[processedData['visiumOtsu']==True] = sampleData['imageDataGray'][processedData['visiumOtsu']==True]
#     # second gaussian is the one used in registration
#     sampleGauss = filters.gaussian(sampleData['imageDataGray'], sigma=5)
#     tissueGauss = np.zeros(sampleGauss.shape)
#     tissueGauss[processedData['visiumOtsu']==True] = sampleGauss[processedData['visiumOtsu']==True]
#     tissueNormalized = cv2.normalize(tissueGauss, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
#     # due to the large size of the tif files, the image is downsampled during the previous step
#     # tissueResized = rescale(tissueNormalized,resolutionRatio)
#     # if rotation < 0:
#     #     tissuePointsRotated = rotateTissuePoints(sampleData, rotation)
#     #     rotation = rotation * -1
#     #     tissueRotated = rotate(tissueResized, rotation, resize=True)
#     #     tissueRotated = tissueRotated[:,::-1]
#     # else:
#     #     tissueRotated = rotate(tissueResized, rotation, resize=True)
#     #     tissuePointsRotated = rotateTissuePoints(visiumData, rotation)
#     processedData['tissueProcessed'] = match_histograms(tissueNormalized, templateData['rightHem'])
#     processedData['tissueProcessed'] = processedData['tissueProcessed'] - processedData['tissueProcessed'].min()
    
#     tissuePointsResized = processedData['tissuePositionsList'] * 0.09

#     # denseMatrix = sampleData["geneMatrix"]
#     # denseMatrix = denseMatrix.todense().astype('float32')
    
#     processedData['geneListMasked'] = sampleData['geneList']
#     processedData['processedTissuePositionList'] = tissuePointsResized
#     if log2normalize==True:
#         processedData['geneMatrixLog2'] = sp_sparse.csc_matrix(np.log2((processedData['geneMatrix'] + 1)))
#         sp_sparse.save_npz(f"{os.path.join(outputPath,processedData['sampleID'])}_tissuePointOrderedGeneMatrixLog2Normalized.npz", processedData['geneMatrixLog2'])
        
#     else:
#         sp_sparse.save_npz(f"{os.path.join(outputPath,processedData['sampleID'])}_tissuePointOrderedGeneMatrix.npz", processedData['geneMatrix'])
        
#     # finiteMin=np.min(countsPerSpotZscore)
#     # finiteMax=np.max(countsPerSpotZscore)
#     # zeroCenteredCmap = mcolors.TwoSlopeNorm(0,vmin=finiteMin, vmax=finiteMax)
#     # plt.imshow( processedVisium['tissueRotated'], cmap='gray')
#     # plt.scatter(processedVisium['tissuePointsResized'][:,0],processedVisium['tissuePointsResized'][:,1], c=np.array(countsPerSpot), alpha=0.8)
#     # plt.title(f"Total gene count per spot for {processedVisium['sampleID']}")
#     # plt.colorbar()
#     # plt.show()
#     # plt.imshow(processedData['tissueProcessed'], cmap='gray')
#     # plt.scatter(tissuePointsResized[:,0],tissuePointsResized[:,1], c=np.array(countsPerSpotZscore), alpha=0.8, cmap='seismic', norm=zeroCenteredCmap, marker='.')
#     # plt.title(f"Z-score of overall gene count per spot for {processedVisium['sampleID']}")
#     # plt.colorbar()
#     # plt.show()
    
#     # write outputs
#     # writes json containing general info and masked gene list
#     processedDataDict = {
#         "sampleID": sampleData['sampleID'],
#         "rotation": int(rotation),
#         "otsuThreshold": float(otsuThreshold),
#         "geneList": processedData['geneList']
#     }
#         # Serializing json
#     # json_object = json.dumps(processedDataDict, indent=4)
     
#     # # Writing to sample.json
#     # with open(f"{processedData['derivativesPath']}/{processedData['sampleID']}_processing_information.json", "w") as outfile:
#     #     outfile.write(json_object)
#     # writes sorted, masked, normalized filtered feature matrix to .npz file
    
#     # writes image for masked greyscale tissue, as well as the processed image that will be used in registration
#     cv2.imwrite(f"{processedData['derivativesPath']}/{processedData['sampleID']}_tissue.png",255*tissue)
#     cv2.imwrite(f"{processedData['derivativesPath']}/{processedData['sampleID']}_tissueProcessed.png",processedData['tissueProcessed'])
    
#     header=['x','y','z','t','label','comment']
#     rowFormat = []
#     # the x and y are swapped between ants and numpy, but this is so far dealt with within the code
#     with open(f"{os.path.join(outputPath,processedData['sampleID'])}_tissuePointsProcessed.csv", 'w', encoding='UTF8') as f:
#         writer = csv.writer(f)
#         writer.writerow(header)
#         for i in range(len(processedData['processedTissuePositionList'])):
#             rowFormat = [processedData['processedTissuePositionList'][i,1]] + [processedData['processedTissuePositionList'][i,0]] + [0] + [0] + [0] + [0]
#             writer.writerow(rowFormat)
#     return processedData

#%% 

templateData = stanly.chooseTemplateSlice(90)
sampleData = stanly.importMerfishData(sourcedata, derivatives)
processedSample = stanly.processMerfishData(sampleData, templateData, 0, derivatives)
sampleRegistered = stanly.runANTsToAllenRegistration(processedSample, templateData)

actSpots = np.array(np.squeeze(sampleRegistered['geneMatrixMasked'].todense()[0,:]), dtype='int32')
plt.imshow(sampleRegistered['tissueRegistered'], cmap='gray')
plt.scatter(sampleRegistered['maskedTissuePositionList'][:,0],sampleRegistered['maskedTissuePositionList'][:,1], c=actSpots, alpha=0.8, cmap='Reds', marker='.')
# plt.title(f'Gene count for {geneName} in {processedSample["sampleID"]}')
plt.show()

actSpots = np.array(np.squeeze(processedSample['geneMatrix'][0,:]), dtype='int32')
plt.imshow(processedSample['tissueProcessed'], cmap='gray')
plt.scatter(processedSample['processedTissuePositionList'][:,0],processedSample['processedTissuePositionList'][:,1], c=actSpots, alpha=0.8, cmap='Reds', marker='.')
# plt.title(f'Gene count for {geneName} in {processedSample["sampleID"]}')
plt.show()