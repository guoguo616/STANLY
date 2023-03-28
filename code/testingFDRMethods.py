#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 27 10:43:35 2022

@author: zjpeters
"""
import os
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import scipy
import csv
import time
import stanly


rawdata, derivatives = stanly.setExperimentalFolder("/home/zjpeters/Documents/visiumalignment")
#%% load experiment of samples that have already been processed and registered
template = stanly.chooseTemplateSlice(70)
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
imageList = [0,1,2,3,4,5,6,7,10,11,12,13,15]

experiment = {'sample-id': np.asarray(sampleList)[imageList],
                    'template-slice': templateList[imageList,0],
                    'rotation': templateList[imageList,1],
                    'experimental-group': templateList[imageList,2]}

processedSamples = {}
totalSpotCount = 0
for actSample in range(len(experiment['sample-id'])):
    sampleProcessed = stanly.loadProcessedSample(os.path.join(derivatives, experiment['sample-id'][actSample]))
    processedSamples[actSample] = sampleProcessed
    totalSpotCount += sampleProcessed['spotCount']
nTotalSamples = len(processedSamples)
spotCountMean = totalSpotCount / nTotalSamples
print(f"Average spot count across {nTotalSamples} samples is {spotCountMean}")

allSamplesToAllen = {}
for actSample in range(len(experiment['sample-id'])):
    sampleRegistered = stanly.loadAllenRegisteredSample(os.path.join(derivatives, experiment['sample-id'][actSample]))
    allSamplesToAllen[actSample] = sampleRegistered


#%% create digital spots and find nearest neighbors
wholeBrainSpotSize = 15
kSpots = 7
templateDigitalSpots = stanly.createDigitalSpots(allSamplesToAllen[4], wholeBrainSpotSize)

allSampleGeneList = allSamplesToAllen[0]['geneListMasked']
for i, regSample in enumerate(allSamplesToAllen):        
    actNN, actCDist = stanly.findDigitalNearestNeighbors(templateDigitalSpots, allSamplesToAllen[i]['maskedTissuePositionList'], kSpots, wholeBrainSpotSize)
    allSamplesToAllen[i]['digitalSpotNearestNeighbors'] = np.asarray(actNN, dtype=int)
    # creates a list of genes present in all samples
    if i == 0:
        continue
    else:
        allSampleGeneList = set(allSampleGeneList) & set(allSamplesToAllen[i]['geneListMasked'])


nDigitalSpots = len(templateDigitalSpots)
nSampleExperimental = sum(experiment['experimental-group'])
nSampleControl = len(experiment['experimental-group']) - nSampleExperimental

#%% first test using Sidak correction

start_time = time.time()
alphaSidak = 1 - np.power((1 - 0.05),(1/len(allSampleGeneList)))

sigGenes = []
sigGenesWithPvals = []
sigGenesWithTstats = []
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
    for actSample in range(len(allSamplesToAllen)):
        try:
            geneIndex = allSamplesToAllen[actSample]['geneListMasked'].index(actGene)
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
        spotThr = 3 #0.05 * nDigitalSpots
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
            fig = plt.figure()
            fig.add_subplot(1,3,1)
            plt.axis('off')
            plt.imshow(allSamplesToAllen[4]['visiumTransformed'],cmap='gray')
            plt.scatter(templateDigitalSpots[:,0],templateDigitalSpots[:,1], c=np.array(meanDigitalControl), alpha=0.8, vmin=0,vmax=maxGeneCount,plotnonfinite=False,cmap='Reds',marker='.')
            plt.title('NSD')

            # display mean gene count for experimental group
            fig.add_subplot(1,3,2)
            plt.axis('off')
            plt.imshow(allSamplesToAllen[4]['visiumTransformed'],cmap='gray')
            expScatter = plt.scatter(templateDigitalSpots[:,0],templateDigitalSpots[:,1], c=np.array(meanDigitalExperimental), alpha=0.8, vmin=0,vmax=maxGeneCount,plotnonfinite=False,cmap='Reds',marker='.')
            plt.title('SD')
            fig.colorbar(expScatter,fraction=0.046, pad=0.04)

            fig.add_subplot(1,3,3)
            plt.axis('off')
            plt.imshow(allSamplesToAllen[4]['visiumTransformed'],cmap='gray')
            tStatScatter = plt.scatter(templateDigitalSpots[:,0],templateDigitalSpots[:,1], c=np.array(actTstats), cmap='seismic',alpha=0.8,vmin=-4,vmax=4,plotnonfinite=False,marker='.')
            plt.title(actGene, style='italic')
            fig.colorbar(tStatScatter,fraction=0.046, pad=0.04)
            plt.savefig(os.path.join(derivatives,f'tStatGeneCount{actGene}SleepDep.png'), bbox_inches='tight', dpi=300)
            plt.show()

timestr = time.strftime("%Y%m%d-%H%M%S")
with open(os.path.join(derivatives,f'listOfSigSleepDepGenesSidakPvalues{timestr}.csv'), 'w', encoding='UTF8') as f:
    writer = csv.writer(f)
    for i in sigGenesWithPvals:
        writer.writerow(i)
        
with open(os.path.join(derivatives,f'listOfSigSleepDepGenesSidakTstatistics{timestr}.csv'), 'w', encoding='UTF8') as f:
    writer = csv.writer(f)
    for i in sigGenesWithTstats:
        writer.writerow(i)
        
print("--- %s seconds ---" % (time.time() - start_time))

###############################################################################
# should repeat the above without the 3 spot threshold
###############################################################################
#%% first test using Benjamani-Hochberg correction 0.05

start_time = time.time()
rankList = np.arange(1,nDigitalSpots+1)
desiredPval = 0.05
bhCorrPval = (rankList/len(allSampleGeneList))*desiredPval

sigGenes = []
sigGenesWithPvals = []
sigGenesWithTstats = []
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
    for actSample in range(len(allSamplesToAllen)):
        try:
            geneIndex = allSamplesToAllen[actSample]['geneListMasked'].index(actGene)
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
    ###########################################################################
    # this will check that at least a certain number of spots show expression #
    # for gene. more about speeding up the overall search than thresholding   #
    ###########################################################################
    checkControlSamples = np.count_nonzero(digitalSamplesControl,axis=1)
    checkExperimentalSamples = np.count_nonzero(digitalSamplesExperimental,axis=1)
    checkAllSamples = checkControlSamples & checkExperimentalSamples > 20
    if sum(checkAllSamples) < 20:
        continue
    else:
        maskedDigitalSamplesControl = np.zeros(digitalSamplesControl.shape)
        maskedDigitalSamplesExperimental = np.zeros(digitalSamplesExperimental.shape)
        maskedDigitalSamplesControl[checkAllSamples,:] = digitalSamplesControl[checkAllSamples,:]
        maskedDigitalSamplesExperimental[checkAllSamples,:] = digitalSamplesExperimental[checkAllSamples,:]
        maskedTtests = []
        allTstats = np.zeros(nDigitalSpots)
        actTtest = scipy.stats.ttest_ind(digitalSamplesExperimental,digitalSamplesControl, axis=1, nan_policy='propagate')
        actTstats = actTtest[0]
        actPvals = actTtest[1]
        sortedPVals = np.sort(actPvals)
        # allPvals.append(actPvals)
        mulCompResults = sortedPVals < bhCorrPval
        spotThr = 3 #0.05 * nDigitalSpots
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
            fig = plt.figure()
            fig.add_subplot(1,3,1)
            plt.axis('off')
            plt.imshow(allSamplesToAllen[4]['visiumTransformed'],cmap='gray')
            plt.scatter(templateDigitalSpots[:,0],templateDigitalSpots[:,1], c=np.array(meanDigitalControl), alpha=0.8, vmin=0,vmax=maxGeneCount,plotnonfinite=False,cmap='Reds',marker='.')
            plt.title('NSD')

            # display mean gene count for experimental group
            fig.add_subplot(1,3,2)
            plt.axis('off')
            plt.imshow(allSamplesToAllen[4]['visiumTransformed'],cmap='gray')
            expScatter = plt.scatter(templateDigitalSpots[:,0],templateDigitalSpots[:,1], c=np.array(meanDigitalExperimental), alpha=0.8, vmin=0,vmax=maxGeneCount,plotnonfinite=False,cmap='Reds',marker='.')
            plt.title('SD')
            fig.colorbar(expScatter,fraction=0.046, pad=0.04)

            fig.add_subplot(1,3,3)
            plt.axis('off')
            plt.imshow(allSamplesToAllen[4]['visiumTransformed'],cmap='gray')
            tStatScatter = plt.scatter(templateDigitalSpots[:,0],templateDigitalSpots[:,1], c=np.array(actTstats), cmap='seismic',alpha=0.8,vmin=-4,vmax=4,plotnonfinite=False,marker='.')
            plt.title(actGene, style='italic')
            fig.colorbar(tStatScatter,fraction=0.046, pad=0.04)
            plt.savefig(os.path.join(derivatives,f'tStatGeneCount{actGene}BenjaminiHochbergSleepDep.png'), bbox_inches='tight', dpi=300)
            plt.show()

timestr = time.strftime("%Y%m%d-%H%M%S")
with open(os.path.join(derivatives,f'listOfSigSleepDepGenesBenjaminiHochbergPvalues{timestr}.csv'), 'w', encoding='UTF8') as f:
    writer = csv.writer(f)
    for i in sigGenesWithPvals:
        writer.writerow(i)
        
with open(os.path.join(derivatives,f'listOfSigSleepDepGenesBenjaminiHochbergTstatistics{timestr}.csv'), 'w', encoding='UTF8') as f:
    writer = csv.writer(f)
    for i in sigGenesWithTstats:
        writer.writerow(i)
        
print("--- %s seconds ---" % (time.time() - start_time))

#%% test using Bonferroni correction 0.05

start_time = time.time()
bonCorrPval = desiredPval/len(allSampleGeneList)

sigGenes = []
sigGenesWithPvals = []
sigGenesWithTstats = []
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
    for actSample in range(len(allSamplesToAllen)):
        try:
            geneIndex = allSamplesToAllen[actSample]['geneListMasked'].index(actGene)
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
    ###########################################################################
    # this will check that at least a certain number of spots show expression #
    # for gene. more about speeding up the overall search than thresholding   #
    ###########################################################################
    checkControlSamples = np.count_nonzero(digitalSamplesControl,axis=1)
    checkExperimentalSamples = np.count_nonzero(digitalSamplesExperimental,axis=1)
    checkAllSamples = checkControlSamples & checkExperimentalSamples > 20
    if sum(checkAllSamples) < 20:
        continue
    else:
        maskedDigitalSamplesControl = np.zeros(digitalSamplesControl.shape)
        maskedDigitalSamplesExperimental = np.zeros(digitalSamplesExperimental.shape)
        maskedDigitalSamplesControl[checkAllSamples,:] = digitalSamplesControl[checkAllSamples,:]
        maskedDigitalSamplesExperimental[checkAllSamples,:] = digitalSamplesExperimental[checkAllSamples,:]
        maskedTtests = []
        allTstats = np.zeros(nDigitalSpots)
        actTtest = scipy.stats.ttest_ind(digitalSamplesExperimental,digitalSamplesControl, axis=1, nan_policy='propagate')
        actTstats = actTtest[0]
        actPvals = actTtest[1]
        sortedPVals = np.sort(actPvals)
        # allPvals.append(actPvals)
        mulCompResults = sortedPVals < bonCorrPval
        spotThr = 3 #0.05 * nDigitalSpots
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
            fig = plt.figure()
            fig.add_subplot(1,3,1)
            plt.axis('off')
            plt.imshow(allSamplesToAllen[4]['visiumTransformed'],cmap='gray')
            plt.scatter(templateDigitalSpots[:,0],templateDigitalSpots[:,1], c=np.array(meanDigitalControl), alpha=0.8, vmin=0,vmax=maxGeneCount,plotnonfinite=False,cmap='Reds',marker='.')
            plt.title('NSD')

            # display mean gene count for experimental group
            fig.add_subplot(1,3,2)
            plt.axis('off')
            plt.imshow(allSamplesToAllen[4]['visiumTransformed'],cmap='gray')
            expScatter = plt.scatter(templateDigitalSpots[:,0],templateDigitalSpots[:,1], c=np.array(meanDigitalExperimental), alpha=0.8, vmin=0,vmax=maxGeneCount,plotnonfinite=False,cmap='Reds',marker='.')
            plt.title('SD')
            fig.colorbar(expScatter,fraction=0.046, pad=0.04)

            fig.add_subplot(1,3,3)
            plt.axis('off')
            plt.imshow(allSamplesToAllen[4]['visiumTransformed'],cmap='gray')
            tStatScatter = plt.scatter(templateDigitalSpots[:,0],templateDigitalSpots[:,1], c=np.array(actTstats), cmap='seismic',alpha=0.8,vmin=-4,vmax=4,plotnonfinite=False,marker='.')
            plt.title(actGene, style='italic')
            fig.colorbar(tStatScatter,fraction=0.046, pad=0.04)
            plt.savefig(os.path.join(derivatives,f'tStatGeneCount{actGene}BonferroniSleepDep.png'), bbox_inches='tight', dpi=300)
            plt.show()

timestr = time.strftime("%Y%m%d-%H%M%S")
with open(os.path.join(derivatives,f'listOfSigSleepDepGenesBonferroniPvalues{timestr}.csv'), 'w', encoding='UTF8') as f:
    writer = csv.writer(f)
    for i in sigGenesWithPvals:
        writer.writerow(i)
        
with open(os.path.join(derivatives,f'listOfSigSleepDepGenesBonferroniTstatistics{timestr}.csv'), 'w', encoding='UTF8') as f:
    writer = csv.writer(f)
    for i in sigGenesWithTstats:
        writer.writerow(i)
        
print("--- %s seconds ---" % (time.time() - start_time))

