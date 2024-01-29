#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""".
Created on Sep 7 2022

@author: zjpeters.
"""
# data being read in includes: json, h5, csv, nrrd, jpg, and svg
import os
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
# import json
import csv
# import cv2
# from glob import glob
import scipy
# import scipy.spatial as sp_spatial
# import scipy.sparse as sp_sparse
import time
import sys
sys.path.insert(0, "/home/zjpeters/Documents/stanly/code")
import stanly
# from scipy.spatial.distance import pdist, squareform, cosine, cdist
# setting up paths
rawdata, derivatives = stanly.setExperimentalFolder("/home/zjpeters/Documents/stanly")

#%% import sample list, location, and degrees of rotation from participants.tsv
#sampleList contains sample ids, templateList contains template slices and degrees of rotation to match
template = stanly.chooseTemplateSlice(70)
sampleList = []
templateList = []
with open(os.path.join(rawdata,"participants.tsv"), newline='') as tsvfile:
    tsvreader = csv.reader(tsvfile, delimiter='\t')
    next(tsvreader)
    for row in tsvreader:
        sampleList.append(row[0])
        templateList.append(row[1:])

templateList = np.array(templateList, dtype='int32')

# run edge detection on annotation image
# start by renumbering annotated version so it doesn't have large numbers

# list of good images
imageList = [0,1,2,3,4,5,6,7,10,11,12,13,15]

experiment = {'sample-id': np.asarray(sampleList)[imageList],
                    'rotation': templateList[imageList,1],
                    'experimental-group': templateList[imageList,2],
                    'flip': templateList[imageList,3]}

processedSamples = {}
totalSpotCount = 0
for actSample in range(len(experiment['sample-id'])):
    sample = stanly.importVisiumData(os.path.join(rawdata, experiment['sample-id'][actSample]))
    flipBool=False
    if experiment['flip'][actSample]==1:
        flipBool=True
    sampleProcessed = stanly.processVisiumData(sample, template, experiment['rotation'][actSample], derivatives, flip=flipBool)
    processedSamples[actSample] = sampleProcessed
    totalSpotCount += sampleProcessed['spotCount']
nTotalSamples = len(processedSamples)
spotCountMean = totalSpotCount / nTotalSamples
print(f"Average spot count across {nTotalSamples} samples is {spotCountMean}")
#%% show expression for 1 gene in all samples
plt.close('all')
actGene = 'Slc1a2'
for i, regSample in enumerate(processedSamples):
    stanly.viewGeneInProcessedVisium(processedSamples[i], actGene)

#%% register all samples to best fit and then to CCF
plt.close('all')
bestSampleToTemplate = stanly.runANTsToAllenRegistration(processedSamples[4], template, hemisphere='rightHem')
#%% 
experimentalResults = {}
for actSample in range(len(processedSamples)):
    sampleRegistered = stanly.runANTsInterSampleRegistration(processedSamples[actSample], processedSamples[4])
    experimentalResults[actSample] = sampleRegistered

allSamplesToAllen = {}
for actSample in range(len(experimentalResults)):
    regSampleToTemplate = stanly.applyAntsTransformations(experimentalResults[actSample], bestSampleToTemplate, template, hemisphere='rightHem')
    allSamplesToAllen[actSample] = regSampleToTemplate


#%% create digital spots for whole slice and find nearest neighbors

wholeBrainSpotSize = 15
kSpots = 7
templateDigitalSpots = stanly.createDigitalSpots(allSamplesToAllen[4], wholeBrainSpotSize)

allSampleGeneList = allSamplesToAllen[0]['geneListMasked']
for i, regSample in enumerate(allSamplesToAllen):        
    actNN, actCDist = stanly.findDigitalNearestNeighbors(templateDigitalSpots, allSamplesToAllen[i]['maskedTissuePositionList'], kSpots, wholeBrainSpotSize)
    allSamplesToAllen[i]['digitalSpotNearestNeighbors'] = np.asarray(actNN, dtype='int32')
    # creates a list of genes present in all samples
    if i == 0:
        continue
    else:
        allSampleGeneList = set(allSampleGeneList) & set(allSamplesToAllen[i]['geneListMasked'])

allSampleGeneListSorted = sorted(allSampleGeneList)

nDigitalSpots = len(templateDigitalSpots)
nSampleExperimental = sum(experiment['experimental-group'])
nSampleControl = len(experiment['experimental-group']) - nSampleExperimental
nGenesInList = len(allSampleGeneList)

for sampleIdx, actSample in enumerate(allSamplesToAllen):
    allSamplesToAllen[sampleIdx]['allSampleGeneList'] = allSampleGeneList 
    sortedIdxList = np.zeros(nGenesInList,dtype='int32')
    for sortedIdx, actGene in enumerate(allSampleGeneList):
        sortedIdxList[sortedIdx] = allSamplesToAllen[sampleIdx]['geneListMasked'].index(actGene)
    allSamplesToAllen[sampleIdx]['geneMatrixMaskedSorted'] = allSamplesToAllen[sampleIdx]['geneMatrixMasked'][sortedIdxList,:].astype('int32')
    allSamplesToAllen[sampleIdx].pop('geneMatrixMasked')
    allSamplesToAllen[sampleIdx].pop('geneListMasked')

#%% first test regional using Sidak correction
plt.close('all')
nameForMask = 'WholeBrain'
start_time = time.time()
desiredPval = 0.05
alphaSidak = 1 - np.power((1 - desiredPval),(1/(len(allSampleGeneList))))
geneList = stanly.loadGeneListFromCsv('/home/zjpeters/Documents/stanly/derivatives/221224/listOfSigSleepDepGenes20221224.csv')

sigGenes = []
sigGenesWithPvals = []
sigGenesWithTstats = []
for nOfGenesChecked,actGene in enumerate(geneList):
    digitalSamplesControl = np.zeros([nDigitalSpots,(nSampleControl * kSpots)])
    digitalSamplesExperimental = np.zeros([nDigitalSpots,(nSampleExperimental * kSpots)])
    startControl = 0
    stopControl = kSpots
    startExperimental = 0
    stopExperimental = kSpots
    nTestedSamples = 0
    nControls = 0
    nExperimentals = 0
    for actSample in range(len(allSamplesToAllen)):
        try:
            geneIndex = list(allSamplesToAllen[actSample]['allSampleGeneList']).index(actGene)
        except(ValueError):
            print(f'{actGene} not in dataset')
            continue

        geneCount = np.zeros([nDigitalSpots,kSpots])
        for spots in enumerate(allSamplesToAllen[actSample]['digitalSpotNearestNeighbors']):
            if np.any(spots[1] < 0):
                geneCount[spots[0]] = np.nan
            else:
                spotij = np.zeros([7,2], dtype=int)
                spotij[:,1] = np.asarray(spots[1], dtype=int)
                spotij[:,0] = geneIndex
                
                geneCount[spots[0]] = allSamplesToAllen[actSample]['geneMatrixMaskedSorted'][spotij[:,0],spotij[:,1]]
                
        spotCount = np.nanmean(geneCount, axis=1)
        nTestedSamples += 1
        if experiment['experimental-group'][actSample] == 0:
            digitalSamplesControl[:,startControl:stopControl] = geneCount
            startControl += kSpots
            stopControl += kSpots
            nControls += 1
        elif experiment['experimental-group'][actSample] == 1:
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
        # testControlSamples = digitalSamplesControl[checkAllSamples,:] 
        # testExperimentalSamples = digitalSamplesExperimental[checkAllSamples,:]
        # testSpotCoordinates = templateDigitalSpots[checkAllSamples,:]
        maskedDigitalSamplesControl = np.zeros(digitalSamplesControl.shape)
        maskedDigitalSamplesExperimental = np.zeros(digitalSamplesExperimental.shape)
        maskedDigitalSamplesControl[checkAllSamples,:] = digitalSamplesControl[checkAllSamples,:]
        maskedDigitalSamplesExperimental[checkAllSamples,:] = digitalSamplesExperimental[checkAllSamples,:]
        maskedTtests = []
        allTstats = np.zeros(nDigitalSpots)
        actTtest = scipy.stats.ttest_ind(digitalSamplesExperimental,digitalSamplesControl, axis=1, nan_policy='propagate')
        actTstats = actTtest[0]
        actPvals = actTtest[1]
        # allPvals.append(actPvals)
        mulCompResults = actPvals < alphaSidak
        # mulCompResults = multipletests(actTtest[1], 0.05, method='bonferroni', is_sorted=False)
        # fdrAlpha = mulCompResults[3].
        
        if sum(mulCompResults) > 0:
            actSigGene = [actGene,sum(mulCompResults)]
            sigGenes.append(actSigGene)
            actSigGeneWithPvals = np.append(actSigGene, actPvals)
            actSigGeneWithTstats = np.append(actSigGene, actTstats)
            sigGenesWithPvals.append(actSigGeneWithPvals)
            sigGenesWithTstats.append(actSigGeneWithTstats)
            maskedDigitalCoordinates = templateDigitalSpots[np.array(mulCompResults)]
            maskedTstats = actTtest[0][mulCompResults]
            maskedDigitalCoordinates = np.array(maskedDigitalCoordinates)
            medianDigitalControl = np.nanmedian(digitalSamplesControl,axis=1)
            medianDigitalExperimental = np.nanmedian(digitalSamplesExperimental,axis=1)
            # meanDigitalControl = scipy.stats.mode(digitalSamplesControl,axis=1)
            # meanDigitalExperimental = scipy.stats.mode(digitalSamplesExperimental,axis=1)
            meanDigitalControl = np.nanmean(digitalSamplesControl,axis=1)
            meanDigitalExperimental = np.nanmean(digitalSamplesExperimental,axis=1)
            finiteMin = np.nanmin(actTtest[0])
            finiteMax = np.nanmax(actTtest[0])
            maxGeneCount = np.nanmax([medianDigitalControl,medianDigitalExperimental])
            # display mean gene count for control group            
            # fig = plt.figure()
            #Plot data
            fig, axs = plt.subplots(1,3)
            
            # axs[0].plot(Y)
            # axs[1].scatter(z['level_1'], z['level_0'],c=z[0])
            conMax = np.nanmax(meanDigitalControl)
            expMax = np.nanmax(meanDigitalExperimental)
            plotMax = np.max([conMax, expMax])
            # fig.add_subplot(1,3,1)
            plt.axis('off')
            fig.suptitle(f'{actGene}', style='italic', y=0.80)
            axs[0].imshow(bestSampleToTemplate['tissueRegistered'],cmap='gray',aspect="equal")
            controlScatter = axs[0].scatter(templateDigitalSpots[:,0],templateDigitalSpots[:,1], c=np.array(meanDigitalControl), alpha=0.5,plotnonfinite=False,cmap='Reds',marker='.', vmin=0, vmax=plotMax, s=5)
            # axs[0].imshow(template['leftHemAnnotEdges'], cmap='gray_r')
            axs[0].set_title('NSD')
            axs[0].axis('off')
            fig.colorbar(controlScatter, fraction=0.046, pad=0.04)
            # display mean gene count for experimental group
            axs[1].imshow(bestSampleToTemplate['tissueRegistered'],cmap='gray',aspect="equal")
            experimentalScatter = axs[1].scatter(templateDigitalSpots[:,0],templateDigitalSpots[:,1], c=np.array(meanDigitalExperimental), alpha=0.5,plotnonfinite=False,cmap='Reds',marker='.', vmin=0, vmax=plotMax, s=5)
            axs[1].set_title('Sleep Dep')
            axs[1].axis('off')
            fig.colorbar(experimentalScatter, fraction=0.046, pad=0.04)
            # display t-statistic for exp > control
            tScatter = axs[2].scatter(templateDigitalSpots[:,0],templateDigitalSpots[:,1], c=np.array(actTstats), cmap='seismic',alpha=0.5,vmin=-4,vmax=4,plotnonfinite=False,marker='.', s=5)
            axs[2].imshow(bestSampleToTemplate['tissueRegistered'],cmap='gray',aspect="equal")
            axs[2].set_title('t-statistic\nSleep Dep > HC')
            axs[2].axis('off')
            fig.colorbar(tScatter, fraction=0.046, pad=0.04)
            fig.tight_layout()
            plt.savefig(os.path.join(derivatives,f'tStatGeneCount{nameForMask}{actGene}SleepDep.png'), bbox_inches='tight', dpi=300)
            plt.show()
            plt.close()

timestr = time.strftime("%Y%m%d-%H%M%S")
with open(os.path.join(derivatives,f'listOfSigSleepDepGenesSidakPvalues_{nameForMask}_{timestr}.csv'), 'w', encoding='UTF8') as f:
    writer = csv.writer(f)
    for i in sigGenesWithPvals:
        writer.writerow(i)
        
with open(os.path.join(derivatives,f'listOfSigSleepDepGenesSidakTstatistics_{nameForMask}_{timestr}.csv'), 'w', encoding='UTF8') as f:
    writer = csv.writer(f)
    for i in sigGenesWithTstats:
        writer.writerow(i)
        
print("--- %s seconds ---" % (time.time() - start_time))


#%% rewriting t stat section to output everything asked for, no matter significance, good for searching a gene list
wholeBrainSpotSize = 15
templateDigitalSpots = stanly.createDigitalSpots(bestSampleToTemplate, wholeBrainSpotSize)
for i, regSample in enumerate(allSamplesToAllen):        
    actNN, actCDist = stanly.findDigitalNearestNeighbors(templateDigitalSpots, allSamplesToAllen[i]['maskedTissuePositionList'], kSpots, wholeBrainSpotSize)
    allSamplesToAllen[i]['digitalSpotNearestNeighbors'] = np.asarray(actNN, dtype=int)
    # creates a list of genes present in all samples
    if i == 0:
        continue
    else:
        allSampleGeneList = set(allSampleGeneList) & set(allSamplesToAllen[i]['geneListMasked'])

genesFromYann = ['Arc', 'Cirbp', 'Rbm3', 'Homer1', 'Pmch', 'Hcrt', 'Ttr', 'Bhlhe41','Marcksl1','Ep300','Nr4a1','Per1','Ube3a','Sh3gl1','Slc9a3r1','Bhlh41', 'MarcksL1', 'Malat1', 'Hart', 'Ep300', 'Ube3a']
start_time = time.time()

nDigitalSpots = len(templateDigitalSpots)

nSampleExperimental = sum(experiment['experimental-group'])
nSampleControl = len(experiment['experimental-group']) - nSampleExperimental

alphaSidak = 1 - np.power((1 - 0.05),(1/nDigitalSpots))
# alphaSidak = 5e-8
# list(allSampleGeneList)[0:1000]
geneList = allSampleGeneList
sigGenes = []
for nOfGenesChecked,actGene in enumerate(genesFromYann):
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
                spotij = np.zeros([7,2], dtype=int)
                spotij[:,1] = np.asarray(spots[1], dtype=int)
                spotij[:,0] = geneIndex
                
                geneCount[spots[0]] = allSamplesToAllen[actSample]['filteredFeatureMatrixMasked'][spotij[:,0],spotij[:,1]]
                
        spotCount = np.nanmean(geneCount, axis=1)
        nTestedSamples += 1
        if experiment['experimental-group'][actSample] == 0:
            digitalSamplesControl[:,startControl:stopControl] = geneCount
            startControl += kSpots
            stopControl += kSpots
            nControls += 1
        elif experiment['experimental-group'][actSample] == 1:
            digitalSamplesExperimental[:,startExperimental:stopExperimental] = geneCount
            startExperimental += kSpots
            stopExperimental += kSpots
            nExperimentals += 1
            
        else:
            continue
    
    
    digitalSamplesControl = np.array(digitalSamplesControl, dtype=float).squeeze()
    digitalSamplesExperimental = np.array(digitalSamplesExperimental, dtype=float).squeeze()

    allTstats = np.zeros(nDigitalSpots)
    allPvals = []
    actTtest = scipy.stats.ttest_ind(digitalSamplesExperimental,digitalSamplesControl, axis=1, nan_policy='propagate')
    actTstats = actTtest[0]
    actPvals = actTtest[1]
    medianDigitalControl = np.nanmedian(digitalSamplesControl,axis=1)
    medianDigitalExperimental = np.nanmedian(digitalSamplesExperimental,axis=1)
    # meanDigitalControl = scipy.stats.mode(digitalSamplesControl,axis=1)
    # meanDigitalExperimental = scipy.stats.mode(digitalSamplesExperimental,axis=1)
    meanDigitalControl = np.nanmean(digitalSamplesControl,axis=1)
    meanDigitalExperimental = np.nanmean(digitalSamplesExperimental,axis=1)

    zeroCenteredCmap = mcolors.TwoSlopeNorm(0,vmin=finiteMin, vmax=finiteMax)
    tTestColormap = zeroCenteredCmap(actTtest[0])
    maxGeneCount = np.nanmax([medianDigitalControl,medianDigitalExperimental])
    # display mean gene count for control group            
    fig = plt.figure()
    fig.add_subplot(1,3,1)
    plt.axis('off')
    plt.imshow(bestSampleToTemplate['visiumTransformed'],cmap='gray')
    plt.scatter(templateDigitalSpots[:,0],templateDigitalSpots[:,1], c=np.array(meanDigitalControl), alpha=0.8, vmin=0,vmax=maxGeneCount,plotnonfinite=False,cmap='Reds',marker='.')
    plt.title('NSD')

    # plt.savefig(os.path.join(derivatives,f'meanGeneCount{actGene}Control.png'), bbox_inches='tight', dpi=300)
    # plt.show()
    # display mean gene count for experimental group
    fig.add_subplot(1,3,2)
    plt.axis('off')
    plt.imshow(bestSampleToTemplate['visiumTransformed'],cmap='gray')
    expScatter = plt.scatter(templateDigitalSpots[:,0],templateDigitalSpots[:,1], c=np.array(meanDigitalExperimental), alpha=0.8, vmin=0,vmax=maxGeneCount,plotnonfinite=False,cmap='Reds',marker='.')
    plt.title('SD')
    fig.colorbar(expScatter,fraction=0.046, pad=0.04)

    fig.add_subplot(1,3,3)
    plt.axis('off')
    plt.imshow(bestSampleToTemplate['visiumTransformed'],cmap='gray')
    tStatScatter = plt.scatter(templateDigitalSpots[:,0],templateDigitalSpots[:,1], c=np.array(actTtest[0]), cmap='seismic',alpha=0.8,vmin=-4,vmax=4,plotnonfinite=False,marker='.')
    plt.title(f't-statistic for {actGene}')
    fig.colorbar(tStatScatter,fraction=0.046, pad=0.04)
    plt.savefig(os.path.join(derivatives,f'tStatGeneCount{actGene}SleepDep.png'), bbox_inches='tight', dpi=300)
    plt.show()
timestr = time.strftime("%Y%m%d-%H%M%S")
print("--- %s seconds ---" % (time.time() - start_time))

