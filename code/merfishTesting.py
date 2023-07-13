#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 14:37:22 2023

@author: zjpeters
"""
import os
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from skimage.transform import rescale, rotate, resize
import itk
import sys
sys.path.insert(0, "/home/zjpeters/rdss_tnj/visiumalignment/code")
import stanly
from glob import glob
from skimage import io, filters, color, feature, morphology
rawdata, derivatives = stanly.setExperimentalFolder("/home/zjpeters/Documents/visiumalignment")
#%% location of merfish csv data 
sourcedata = os.path.join('/','home','zjpeters','Documents','visiumalignment','sourcedata','merscopedata')
locOfCellByGeneCsv = os.path.join(sourcedata,'datasets_mouse_brain_map_BrainReceptorShowcase_Slice1_Replicate1_cell_by_gene_S1R1.csv')
locOfCellMetadataCsv = os.path.join(sourcedata,'datasets_mouse_brain_map_BrainReceptorShowcase_Slice1_Replicate1_cell_metadata_S1R1.csv')
tifFilename = os.path.join(sourcedata,'datasets_mouse_brain_map_BrainReceptorShowcase_Slice1_Replicate1_images_mosaic_DAPI_z0.tif')
# load data as pandas dataframe and extract list of genes
cellByGene = pd.read_csv(locOfCellByGeneCsv)
cellMetadata = pd.read_csv(locOfCellMetadataCsv)
geneList = cellByGene.columns[1:]

#%% display all genes
for actGene in geneList:
    plt.scatter(cellMetadata.center_x,cellMetadata.center_y, c=cellByGene[actGene], marker='.', cmap='Reds')
    plt.title(actGene)
    plt.gca().invert_yaxis()
    plt.show()

#%% new functions

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

# starting from the importVisiumData and processVisiumData function, create merfish equivalents
# expected merfish data includes:
# 1. image data, needs downsampled due to very high resolution
# 2. cell_by_gene.csv containing cell index and barcode in first two columns followed by columns of rna expression per gene per cell
# 3. cell_metadata.csv containing cell index, barcode, fov, volume, center_x, center_y, min_x, min_y, max_x, max_y
# 

#%% using itk to load and downsample tiff

stanly.downsampleMerfishTiff(tifFilename, os.path.join('/','home','zjpeters','Documents','visiumalignment','derivatives','downsampledMerfish.tif'), scale=0.01)
testSampleImg = io.imread(os.path.join('/','home','zjpeters','Documents','visiumalignment','derivatives','downsampledMerfish.tif'), as_gray=True)

# need to look back over code for downsampling to see why scale of 0.01 requires multiplying by scale * 9, like below
newX = cellMetadata.center_x * 0.09
newY = cellMetadata.center_y * 0.09
plt.imshow(testSampleImg, cmap='gray')
plt.scatter(newX,newY, c=cellByGene['Drd2'], marker='.', cmap='Reds', alpha=0.4)
plt.show()

def importMerfishData(sampleFolder, outputPath):
    # 
    sampleData = {}
    if os.path.exists(os.path.join(sampleFolder)):
        # spatialFolder = os.path.join(sampleFolder)
        try:
            # need to check how the cell by gene file is usually output/named
            os.path.isfile(glob(os.path.join(sampleFolder, '*cell_by_gene_*.csv'))[0])
            dataFolder = sampleFolder
        except IndexError:
            print("Something is wrong!")
            # os.path.isfile(glob(os.path.join(spatialFolder, '*filtered_feature_bc_matrix.h5'))[0])
            # dataFolder = spatialFolder
        # dataFolder = os.path.join(sampleFolder)
    else:
        print(f"{sampleFolder} not found!")
    # need to look for standard outputname of tiff file for registration
    
    originalImagePath =  glob(os.path.join(dataFolder,"*_images_mosaic_DAPI_z0.tif"))[0]
    downsampledImagePath = os.path.splitext(originalImagePath)[0]
    downsampledImagePath = downsampledImagePath + "_downsampled.tif"
    # check if image has already been downsampled
    if any(glob(downsampledImagePath)):
        print(f"Loading previously downsampled image from {sampleFolder}")
    else:
        print("Downsampling high resolution image to 10 micron resolution")
        downsampleMerfishTiff(originalImagePath, downsampledImagePath, scale=0.01)
    sampleData['imageData'] = io.imread(downsampledImagePath)
    # need to convert into 0-1 
    sampleImageNorm = (sampleData['imageData'] - np.min(sampleData['imageData']))/(np.max(sampleData['imageData']) - np.min(sampleData['imageData']))
    sampleData['imageDataGray'] = (sampleData['imageData'] - np.min(sampleData['imageData']))/(np.max(sampleData['imageData']) - np.min(sampleData['imageData']))
    sampleData['sampleID'] = sampleFolder.rsplit(sep='/',maxsplit=1)[-1]
    ###########################################################################
    # stopped here
    ###########################################################################
    tissuePositionsList = []
    tissueSpotBarcodes = []
    with open(os.path.join(dataFolder,"tissue_positions_list.csv"), newline='') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',')
        for row in csvreader:
            # checks for in tissue spots
            if row[1] == '1':
                tissueSpotBarcodes.append(row[0])
                tissuePositionsList.append(row[1:])
    tissuePositionsList = np.array(tissuePositionsList, dtype='float32')
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

def processMerfishData(sampleData, templateData, rotation, outputFolder, log2normalize=True):
    processedData = {}
    # the sampleID might have issues on non unix given the slash direction, might need to fix
    processedData['sampleID'] = sampleData['sampleID']
    outputPath = os.path.join(outputFolder, visiumData['sampleID'])
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
    processedVisium['tissueProcessed'] = match_histograms(tissueRotated, templateData['rightHem'])
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
    filteredFeatureMatrixGeneList = np.array(filteredFeatureMatrixGeneList)
    processedVisium['geneListMasked'] = filteredFeatureMatrixGeneList[geneMask].tolist()
    orderedDenseMatrixSpotMasked = orderedDenseMatrix[:,spotMask]
    orderedDenseMatrixSpotMasked = orderedDenseMatrixSpotMasked[geneMask,:]
    processedVisium['spotCount'] = orderedDenseMatrixSpotMasked.shape[1]
    print(f"{processedVisium['sampleID']} has {processedVisium['spotCount']} spots")
    processedVisium['processedTissuePositionList'] = tissuePointsResized[spotMask,:]
    # processedVisium['filteredFeatureMatrixOrdered'] = sp_sparse.csc_matrix(orderedDenseMatrixSpotMasked)
    if log2normalize==True:
        processedVisium['filteredFeatureMatrixLog2'] = sp_sparse.csc_matrix(np.log2((orderedDenseMatrixSpotMasked + 1)))
        sp_sparse.save_npz(f"{os.path.join(outputPath,processedVisium['sampleID'])}_tissuePointOrderedFeatureMatrixLog2Normalized.npz", processedVisium['filteredFeatureMatrixLog2'])
        
    else:
        processedVisium['filteredFeatureMatrix'] = sp_sparse.csc_matrix(orderedDenseMatrixSpotMasked)
        sp_sparse.save_npz(f"{os.path.join(outputPath,processedVisium['sampleID'])}_tissuePointOrderedFeatureMatrix.npz", processedVisium['filteredFeatureMatrix'])
        
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

