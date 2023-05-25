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


#%% digital spot creation
# so far testing has been best at a spot diameter of 18 pixels
regionalSpotSize = 15
kSpots = 7

desiredRegion = 'Hippocampal region'
regionMask = stanly.createRegionalMask(template, desiredRegion)
regionMaskDigitalSpots = stanly.createRegionalDigitalSpots(regionMask, regionalSpotSize)

allSampleGeneList = allSamplesToAllen[0]['geneListMasked']
for i, regSample in enumerate(['Arc']):        
    actNN, actCDist = stanly.findDigitalNearestNeighbors(regionMaskDigitalSpots, allSamplesToAllen[i]['maskedTissuePositionList'], kSpots, regionalSpotSize)
    allSamplesToAllen[i]['digitalSpotNearestNeighbors'] = np.asarray(actNN, dtype=int)
    # creates a list of genes present in all samples
    if i > 0:
        allSampleGeneList = set(allSampleGeneList) & set(allSamplesToAllen[i]['geneListMasked'])

nDigitalSpots = len(regionMaskDigitalSpots)
#%% first test using Sidak correction
experimentalMask = experiment['experimental-group']
start_time = time.time()
stanly.runTTest(allSamplesToAllen, experimentalMask, allSampleGeneList)
print("--- %s seconds ---" % (time.time() - start_time))

#%% first test using Benjamani-Hochberg correction 0.05

start_time = time.time()
stanly.runTTest(allSamplesToAllen, experimentalMask, allSampleGeneList, fdr='benjamini-hochberg')
print("--- %s seconds ---" % (time.time() - start_time))

#%% test using Bonferroni correction 0.05

start_time = time.time()
stanly.runTTest(allSamplesToAllen, experimentalMask, allSampleGeneList, fdr='bonferroni')
print("--- %s seconds ---" % (time.time() - start_time))

