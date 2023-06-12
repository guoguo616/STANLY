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
import sys
sys.path.insert(0, "/home/zjpeters/rdss_tnj/visiumalignment/code")
import stanly


rawdata, derivatives = stanly.setExperimentalFolder("/home/zjpeters/rdss_tnj/visiumalignment")
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

# run edge detection on annotation image
# start by renumbering annotated version so it doesn't have large numbers

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

experimentalResults = {}
for actSample in range(len(experiment['sample-id'])):
    sampleRegistered = stanly.loadAllenRegisteredSample(os.path.join(derivatives, experiment['sample-id'][actSample]))
    experimentalResults[actSample] = sampleRegistered


#%% create digital spots for whole slice and find nearest neighbors
# ONLY RUN ONE OF THE FOLLOWING TWO SECTIONS, OTHERWISE
wholeBrainSpotSize = 15
kSpots = 7
templateDigitalSpots = stanly.createDigitalSpots(experimentalResults[4], wholeBrainSpotSize)

allSampleGeneList = experimentalResults[0]['geneListMasked']
for i, regSample in enumerate(experimentalResults):        
    actNN, actCDist = stanly.findDigitalNearestNeighbors(templateDigitalSpots, experimentalResults[i]['maskedTissuePositionList'], kSpots, wholeBrainSpotSize)
    experimentalResults[i]['digitalSpotNearestNeighbors'] = np.asarray(actNN, dtype=int)
    # creates a list of genes present in all samples
    if i == 0:
        continue
    else:
        allSampleGeneList = set(allSampleGeneList) & set(experimentalResults[i]['geneListMasked'])


nDigitalSpots = len(templateDigitalSpots)
nSampleExperimental = sum(experiment['experimental-group'])
nSampleControl = len(experiment['experimental-group']) - nSampleExperimental
nGenesInList = len(allSampleGeneList)
#%% sort gene lists using allSampleGeneList
# in order to keep the kernel from restarting, deleted the unmasked matrix and gene list
for sampleIdx, actSample in enumerate(experimentalResults):
    experimentalResults[sampleIdx]['allSampleGeneList'] = allSampleGeneList 
    sortedIdxList = np.zeros(nGenesInList,dtype=int)
    for sortedIdx, actGene in enumerate(allSampleGeneList):
        sortedIdxList[sortedIdx] = experimentalResults[sampleIdx]['geneListMasked'].index(actGene)
    experimentalResults[sampleIdx]['filteredFeatureMatrixMaskedSorted'] = experimentalResults[sampleIdx]['filteredFeatureMatrixMasked'][sortedIdxList,:]
    experimentalResults[sampleIdx].pop('filteredFeatureMatrixMasked')
    experimentalResults[sampleIdx].pop('geneListMasked')
        # experimentalResults[i]['geneListSortedForGroup'][sortedIdx]

#%% test cosine similarity/spectral clustering in single sample

## could try making 2 column list where column 0 is sample #, column 1 is spot #
## this would give nTotalSamples*kSpots rows  

digitalSpotEdges = []#np.zeros([nDigitalSpots, nTotalSamples*kSpots])
for sampleIdx, actSample in enumerate(experimentalResults):
    if sampleIdx == 0:
        digitalSpotEdges = experimentalResults[sampleIdx]['digitalSpotNearestNeighbors']
    else:
        np.append(digitalSpotEdges,experimentalResults[sampleIdx]['digitalSpotNearestNeighbors'], axis=1)

for actDigitalSpot in range(nDigitalSpots):
    print(actDigitalSpot)
    for sampleIdx, actSample in enumerate(experimentalResults):
        print(experimentalResults[0]['digitalSpotNearestNeighbors'][actDigitalSpot,:])
    for actComp in experimentalResults[0]['digitalSpotNearestNeighbors'][actDigitalSpot,:]:
        x = np.sum(np.dot(experimentalResults[0]['filteredFeatureMatrixMaskedSorted'][:,actComp], experimentalResults[0]['filteredFeatureMatrixMaskedSorted'][:,experimentalResults[0]['digitalSpotNearestNeighbors'][actDigitalSpot,:]]))
    
    
#%%    
start_time = time.time()
desiredPval = 0.05
alphaSidak = 1 - np.power((1 - desiredPval),(1/(len(allSampleGeneList))))
geneList = stanly.loadGeneListFromCsv('/home/zjpeters/rdss_tnj/visiumalignment/derivatives/221224/listOfSigSleepDepGenes20221224.csv')

sigGenes = []
sigGenesWithPvals = []
sigGenesWithTstats = []
# since we've sorted and masked the gene lists for all samples, no need to search list for indices
for geneIndex,actGene in enumerate(allSampleGeneList):
    digitalSamplesControl = np.zeros([nDigitalSpots,(nSampleControl * kSpots)])
    digitalSamplesExperimental = np.zeros([nDigitalSpots,(nSampleExperimental * kSpots)])
    startControl = 0
    stopControl = kSpots
    startExperimental = 0
    stopExperimental = kSpots
    nTestedSamples = 0
    nControls = 0
    nExperimentals = 0
    
    geneCount = np.zeros([nDigitalSpots,kSpots])
    for spots in enumerate(experimentalResults[actSample]['digitalSpotNearestNeighbors']):
        # checks for -9999 in NN to convert to nan
        if np.any(spots[1] < 0):
            geneCount[spots[0]] = np.nan
        else:
            spotij = np.zeros([kSpots,2], dtype=int)
            spotij[:,1] = np.asarray(spots[1], dtype=int)
            spotij[:,0] = geneIndex
            
            geneCount[spots[0]] = experimentalResults[actSample]['filteredFeatureMatrixMasked'][:,spotij[:,1]]
            
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
    checkAllSamples = checkControlSamples & checkExperimentalSamples > 0
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
            
            # fig.add_subplot(1,3,1)
            plt.axis('off')
            axs[0].imshow(experimentalResults[4]['visiumTransformed'],cmap='gray',aspect="equal")
            axs[0].scatter(templateDigitalSpots[:,0],templateDigitalSpots[:,1], c=np.array(meanDigitalControl), vmin=0,vmax=3,plotnonfinite=False,cmap='Reds',marker='.')
            # axs[0].imshow(template['leftHem'], cmap='gray')
            axs[0].set_title('NSD')
            axs[0].axis('off')
            # display mean gene count for experimental group
            # fig.add_subplot(1,3,2)
            # plt.axis('off')
            axs[1].imshow(experimentalResults[4]['visiumTransformed'],cmap='gray',aspect="equal")
            axs[1].scatter(templateDigitalSpots[:,0],templateDigitalSpots[:,1], c=np.array(meanDigitalExperimental), vmin=0,vmax=3,plotnonfinite=False,cmap='Reds',marker='.')
            # axs[1].imshow(template['leftHem'], cmap='gray')
            axs[1].set_title('SD')
            axs[1].axis('off')
            # plt.colorbar(scatterBar,ax=axs[1])
    
            # fig.add_subplot(1,3,3)
            # plt.axis('off')
            # axs[2].imshow(experimentalResults[4]['visiumTransformed'],cmap='gray',aspect="equal")
            axs[2].scatter(templateDigitalSpots[:,0],templateDigitalSpots[:,1], c=np.array(actTstats), cmap='seismic', vmin=-4,vmax=4,plotnonfinite=False,marker='.')
            axs[2].imshow(experimentalResults[4]['visiumTransformed'],cmap='gray')
            axs[2].set_title(actGene, style='italic')
            axs[2].axis('off')
            # plt.colorbar(tBar,ax=axs[2],fraction=0.046, pad=0.04)
            plt.savefig(os.path.join(derivatives,f'tStatGeneCount{actGene}SleepDep.png'), bbox_inches='tight', dpi=300)
            plt.show()

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

#%% show overlay of images with template

plt.imshow(template['leftHem'])
plt.imshow(experimentalResults[4]['visiumTransformed'], cmap='gray',alpha=0.6)
plt.axis('off')
plt.show()
#%% show overlay of images with template

plt.imshow(experimentalResults[4]['visiumTransformed'])
plt.imshow(experimentalResults[5]['visiumTransformed'], cmap='gray',alpha=0.6)
plt.axis('off')
plt.show()
