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
import scipy.spatial as sp_spatial
import scipy.sparse as sp_sparse
import csv
import time
import sys
sys.path.insert(0, "/home/zjpeters/Documents/visiumalignment/code")
import stanly
from sklearn.cluster import KMeans

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
    experimentalResults[actSample]['experimentalStatus'] = experiment['experimental-group'][actSample]


#%% create digital spots for whole slice and find nearest neighbors
# ONLY RUN ONE OF THE FOLLOWING TWO SECTIONS, OTHERWISE
# wholeBrainSpotSize = 15
# kSpots = 7
# templateDigitalSpots = stanly.createDigitalSpots(experimentalResults[4], wholeBrainSpotSize)


for i, regSample in enumerate(experimentalResults):        
    # actNN, actCDist = stanly.findDigitalNearestNeighbors(templateDigitalSpots, experimentalResults[i]['maskedTissuePositionList'], kSpots, wholeBrainSpotSize)
    # experimentalResults[i]['digitalSpotNearestNeighbors'] = np.asarray(actNN, dtype=int)
    # creates a list of genes present in all samples
    if i == 0:
        allSampleGeneList = experimentalResults[0]['geneListMasked']
    else:
        allSampleGeneList = set(allSampleGeneList) & set(experimentalResults[i]['geneListMasked'])


# nDigitalSpots = len(templateDigitalSpots)
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

#%% perform spectral clustering on groups using locations in registered coordinates
# could consider breaking control and experimental into separate sections to save on memory
nControls = 0
nExperimentals = 0
kNN = 12
# 1. generate a list of all coordinates 
for sampleIdx, actSample in enumerate(experimentalResults):
    if experimentalResults[actSample]['experimentalStatus'] == 0:
        if nControls == 0:
            allCoordinatesControl = experimentalResults[actSample]['maskedTissuePositionList']
            allSampleSpotIdxIControl = np.repeat(actSample, experimentalResults[actSample]['maskedTissuePositionList'].shape[0])
            allSampleSpotIdxJControl = np.array(range(experimentalResults[actSample]['maskedTissuePositionList'].shape[0]))
        else:
            allCoordinatesControl = np.append(allCoordinatesControl,experimentalResults[actSample]['maskedTissuePositionList'], axis=0)
            allSampleSpotIdxIControl = np.append(allSampleSpotIdxIControl,np.repeat(actSample, experimentalResults[actSample]['maskedTissuePositionList'].shape[0]), axis=0)
            allSampleSpotIdxJControl = np.append(allSampleSpotIdxJControl,np.array(range(experimentalResults[actSample]['maskedTissuePositionList'].shape[0])), axis=0)
        nControls += 1
    # elif experimentalResults[actSample]['experimentalStatus'] == 1:
    #     if nExperimentals == 0:
    #         allCoordinatesExperimental = experimentalResults[actSample]['maskedTissuePositionList']
    #         allSampleSpotIdxIExperimental = np.repeat(actSample, experimentalResults[actSample]['maskedTissuePositionList'].shape[0])
    #         allSampleSpotIdxJExperimental = np.array(range(experimentalResults[actSample]['maskedTissuePositionList'].shape[0]))
    #     else:
    #         allCoordinatesExperimental = np.append(allCoordinatesExperimental,experimentalResults[actSample]['maskedTissuePositionList'], axis=0)
    #         allSampleSpotIdxIExperimental = np.append(allSampleSpotIdxIExperimental,np.repeat(actSample, experimentalResults[actSample]['maskedTissuePositionList'].shape[0]), axis=0)
    #         allSampleSpotIdxJExperimental = np.append(allSampleSpotIdxJExperimental, np.array(range(experimentalResults[actSample]['maskedTissuePositionList'].shape[0])),axis=0)
    #     nExperimentals += 1
    
# 2. calculate nearest neighbors and select for top kNN
nnControl = sp_spatial.distance.cdist(allCoordinatesControl, allCoordinatesControl, 'euclidean')
# nnControlSortedDist = np.sort(nnControl, axis=1)[:,1:kNN+1]
nnControlSortedIdx = np.argsort(nnControl, axis=1)[:,1:kNN+1]
del(nnControl)
# nnExperimental = sp_spatial.distance.cdist(allCoordinatesExperimental, allCoordinatesExperimental, 'euclidean')
# nnExperimentalSortedDist = np.sort(nnExperimental, axis=1)[:,1:kNN+1]
# nnExperimentalSortedIdx = np.argsort(nnExperimental, axis=1)[:,1:kNN+1]
# del(nnExperimental)

# spotAdjacencyDataControl = []
# for I, nnRow in enumerate(nnControlSortedIdx):
#     for J in nnRow:
#         spotAdjacencyDataControl = spotAdjacencyDataControl.append(np.dot(experimentalResults[allSampleSpotIdxControl[I][:,I],digitalSamplesExperimental[:,J]) / (np.linalg.norm(digitalSamplesExperimental[:,I])*np.linalg.norm(digitalSamplesExperimental[:,J]))
    
# create sparse degree and adjacency matrix
# D = sp_sparse.coo_matrix()
# 3. using column numbers and cosine sim, populate sparse matrix to use for spectral embedding
# 3a. this will require: data, row, and column variables as input for sparse matrix function
#%% remove duplicates from nearest neighbor list
nnEdgeList = np.transpose([np.repeat(np.array(range(nnControlSortedIdx.shape[0])), kNN, axis=0).transpose(), nnControlSortedIdx.flatten().transpose()])
nnEdgeList = np.unique(np.sort(nnEdgeList, axis=1), axis=0)
#############################################################################
#%% takes a long time, if already run just load the adjacency data from csv
#############################################################################
start_time = time.time()
adjacencyDataControl = []
for i, j in nnEdgeList:
    I = experimentalResults[allSampleSpotIdxIControl[i]]['filteredFeatureMatrixMaskedSorted'][:,allSampleSpotIdxJControl[i]].todense()
    J = experimentalResults[allSampleSpotIdxIControl[j]]['filteredFeatureMatrixMaskedSorted'][:,allSampleSpotIdxJControl[j]].todense()
    cs = np.sum(np.dot(I,J.transpose())) / np.sum((np.linalg.norm(I)*np.linalg.norm(J)))
    adjacencyDataControl.append(cs)
        # print(cs)
print("--- %s seconds ---" % (time.time() - start_time))  

with open(os.path.join(derivatives,f'adjacencyDataForControl.csv'), 'w', encoding='UTF8') as f:
    writer = csv.writer(f)
    # for i in np.array(adjacencyDataControl):
    writer.writerow(adjacencyDataControl)          
    # spotDiagMatrixControl = np.diag(sum(spotAdjacencyMatrixControl))
    # spotDiagMatrixExperimental = np.diag(sum(spotAjacencyMatrixExperimental))
    # laplacianControl = spotDiagMatrixControl - spotAdjacencyMatrixControl
    # laplacianExperimental = spotDiagMatrixExperimental -spotAdjacencyMatrixExperimental
    
    
#%%
adjacencyDataControl = []
# with open(os.path.join(derivatives,f'adjacencyDataForControl.csv'), 'w', encoding='UTF8') as f:
with open(os.path.join(derivatives,"adjacencyDataForControl.csv"), newline='') as csvfile:
    tsvreader = csv.reader(csvfile, delimiter=',')
    for row in tsvreader:
        adjacencyDataControl = row
#%%
W = np.zeros([allSampleSpotIdxIControl.shape[0],allSampleSpotIdxIControl.shape[0]])
for idx, actCS in enumerate(adjacencyDataControl):
    W[nnEdgeList[idx,0],nnEdgeList[idx,1]] = actCS
    W[nnEdgeList[idx,1],nnEdgeList[idx,0]] = actCS

W = (W - W.min())/(W.max() - W.min())
D = np.diag(sum(W))
L = D - W
L = sp_sparse.coo_matrix(L)
eigval,eigvec = sp_sparse.linalg.eigs(L, k=300)
clusters = KMeans(n_clusters=80, random_state=0).fit(np.real(eigvec))
plt.scatter(allCoordinatesControl[0:2000,0], allCoordinatesControl[0:2000,1],c=clusters.labels_[0:2000])
#%%
    
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
            axs[0].imshow(experimentalResults[4]['tissueRegistered'],cmap='gray',aspect="equal")
            axs[0].scatter(templateDigitalSpots[:,0],templateDigitalSpots[:,1], c=np.array(meanDigitalControl), vmin=0,vmax=3,plotnonfinite=False,cmap='Reds',marker='.')
            # axs[0].imshow(template['leftHem'], cmap='gray')
            axs[0].set_title('NSD')
            axs[0].axis('off')
            # display mean gene count for experimental group
            # fig.add_subplot(1,3,2)
            # plt.axis('off')
            axs[1].imshow(experimentalResults[4]['tissueRegistered'],cmap='gray',aspect="equal")
            axs[1].scatter(templateDigitalSpots[:,0],templateDigitalSpots[:,1], c=np.array(meanDigitalExperimental), vmin=0,vmax=3,plotnonfinite=False,cmap='Reds',marker='.')
            # axs[1].imshow(template['leftHem'], cmap='gray')
            axs[1].set_title('SD')
            axs[1].axis('off')
            # plt.colorbar(scatterBar,ax=axs[1])
    
            # fig.add_subplot(1,3,3)
            # plt.axis('off')
            # axs[2].imshow(experimentalResults[4]['tissueRegistered'],cmap='gray',aspect="equal")
            axs[2].scatter(templateDigitalSpots[:,0],templateDigitalSpots[:,1], c=np.array(actTstats), cmap='seismic', vmin=-4,vmax=4,plotnonfinite=False,marker='.')
            axs[2].imshow(experimentalResults[4]['tissueRegistered'],cmap='gray')
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

plt.imshow(template['rightHem'])
plt.imshow(experimentalResults[4]['tissueRegistered'], cmap='gray',alpha=0.6)
plt.axis('off')
plt.show()
#%% show overlay of images with template

plt.imshow(experimentalResults[4]['tissueRegistered'])
plt.imshow(experimentalResults[5]['tissueRegistered'], cmap='gray',alpha=0.6)
plt.axis('off')
plt.show()
#%%    
# start_time = time.time()
# desiredPval = 0.05
# alphaSidak = 1 - np.power((1 - desiredPval),(1/(len(allSampleGeneList))))
# geneList = stanly.loadGeneListFromCsv('/home/zjpeters/rdss_tnj/visiumalignment/derivatives/221224/listOfSigSleepDepGenes20221224.csv')

# sigGenes = []
# sigGenesWithPvals = []
# sigGenesWithTstats = []
# since we've sorted and masked the gene lists for all samples, no need to search list for indices
