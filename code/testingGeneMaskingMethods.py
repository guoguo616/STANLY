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
sys.path.insert(0, "/home/zjpeters/Documents/visiumalignment/code")
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

#%% visualize single gene
actGene = 'Fezf2'
for i, regSample in enumerate(processedSamples):
    stanly.viewGeneInProcessedVisium(processedSamples[i], actGene)
    
#%% create digital spots for whole slice and find nearest neighbors
# ONLY RUN ONE OF THE FOLLOWING TWO SECTIONS, OTHERWISE

# allSampleGeneList = experimentalResults[0]['geneListMasked']
# for i, regSample in enumerate(experimentalResults):        
#     actNN, actCDist = stanly.findDigitalNearestNeighbors(templateDigitalSpots, experimentalResults[i]['maskedTissuePositionList'], kSpots, wholeBrainSpotSize)
#     experimentalResults[i]['digitalSpotNearestNeighbors'] = np.asarray(actNN, dtype=int)
#     # creates a list of genes present in all samples
#     if i == 0:
#         continue
#     else:
#         allSampleGeneList = set(allSampleGeneList) & set(experimentalResults[i]['geneListMasked'])


# nDigitalSpots = len(templateDigitalSpots)
# nSampleExperimental = sum(experiment['experimental-group'])
# nSampleControl = len(experiment['experimental-group']) - nSampleExperimental


#%% This runs a sidak corrected t-test on all spots expressing Tph2

maskingGene = 'Fezf2'
# allSampleGeneList = processedSamples[0]['geneListMasked']
for i, regSample in enumerate(processedSamples):
    processedSamples[i]['geneMaskedSpots'],processedSamples[i]['geneMaskedTissuePositions'] = stanly.selectSpotsWithGene(processedSamples[i], maskingGene)
    print(processedSamples[i]['geneMaskedSpots'].shape)
    if i == 0:
        allSampleGeneList = processedSamples[0]['geneListMasked']
    else:
        allSampleGeneList = set(allSampleGeneList) & set(processedSamples[i]['geneListMasked'])

alphaSidak = 1 - np.power((1 - 0.1),(1/(len(allSampleGeneList)*spotCountMean)))
start_time = time.time()
sigGenes = [['Gene Name','Mean of Control','SEM of Control','Mean of Experimental','SEM of Experimental','t-statistic','p-value']]
for nOfGenesChecked,actGene in enumerate(allSampleGeneList):
    digitalSamplesControl = []
    digitalSamplesExperimental = []
    nTestedSamples = 0
    nControls = 0
    nExperimentals = 0
    for actSample in range(nTotalSamples):
        try:
            geneIndex = processedSamples[actSample]['geneListMasked'].index(actGene)
            nTestedSamples += 1
            if experiment['experimental-group'][actSample] == 0:
                digitalSamplesControl = np.append(digitalSamplesControl, np.squeeze(np.array(processedSamples[actSample]['geneMaskedSpots'][geneIndex,:])))
                nControls += 1
            elif experiment['experimental-group'][actSample] == 1:
                digitalSamplesExperimental = np.append(digitalSamplesExperimental, np.squeeze(np.array(processedSamples[actSample]['geneMaskedSpots'][geneIndex,:])))
                nExperimentals += 1
            else:
                continue
        except(ValueError):
            print(f'{actGene} not in dataset')
            continue


    actTtest = scipy.stats.ttest_ind(digitalSamplesExperimental,digitalSamplesControl, nan_policy='propagate')
    actTstats = actTtest[0]
    actPvals = actTtest[1]
    # mulCompResults = actPvals < alphaSidak
    # spotThr = 0 #0.05 * nDigitalSpots
    if actPvals < alphaSidak:
        maskedTstats = actTtest[0]
        medianDigitalControl = np.nanmedian(digitalSamplesControl)
        medianDigitalExperimental = np.nanmedian(digitalSamplesExperimental)
        meanDigitalControl = np.nanmean(digitalSamplesControl)
        meanDigitalExperimental = np.nanmean(digitalSamplesExperimental)
        semControl = scipy.stats.sem(digitalSamplesControl)
        semExperimental = scipy.stats.sem(digitalSamplesExperimental)
        sigGenes.append([actGene,meanDigitalControl,semControl,meanDigitalExperimental,semExperimental,actTtest[0],actTtest[1]])
        geneMax = np.nanmax(np.append(digitalSamplesControl,digitalSamplesExperimental))
        # display mean gene count for control group            
        xPos = np.arange(2)
        fig,ax = plt.subplots()
        ax.bar(xPos,[meanDigitalControl,meanDigitalExperimental], yerr=[semControl,semExperimental])
        ax.set_ylabel('Log 2 normalized gene count')
        ax.set_xticks(xPos)
        ax.set_xticklabels(['SD','NSD'])
        ax.set_title(f'Mean log 2 normalized gene expression for {actGene}')
        plt.savefig(os.path.join('/','media','zjpeters','Samsung_T5','marcinkiewcz','lkolling','derivatives',f'meanLog2GeneExpression{actGene}.png'))
        plt.show()
        # for i in range(nTotalSamples):
        #     displayGeneIndex = processedSamples[i]['geneListMasked'].index(actGene)
        #     plt.imshow(processedSamples[i]['tissueProcessed'], cmap='gray')
        #     plt.scatter(processedSamples[i]['geneMaskedTissuePositions'][:,0],processedSamples[i]['geneMaskedTissuePositions'][:,1], c=np.array(processedSamples[i]['geneMaskedSpots'][displayGeneIndex,:]), alpha=0.8, cmap='Reds', marker='.')
        #     plt.title(f'Gene count for {actGene} in {processedSamples[i]["sampleID"]}')
        #     plt.colorbar()
        #     # plt.savefig(os.path.join(derivatives,f'geneCount{geneName}{processedSample["sampleID"]}Registered.png'), bbox_inches='tight', dpi=300)
        #     plt.show()

timestr = time.strftime("%Y%m%d-%H%M%S")
with open(os.path.join(derivatives,f'listOfSigGenes_{maskingGene}_{timestr}.csv'), 'w', encoding='UTF8') as f:
    writer = csv.writer(f)
    # for i in sigGenes:
    writer.writerows(sigGenes)
    # writer.writerows((gene,) for gene in sigGenes)
print("--- %s seconds ---" % (time.time() - start_time))

