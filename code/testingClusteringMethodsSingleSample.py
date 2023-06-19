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
sys.path.insert(0, "/home/zjpeters/rdss_tnj/visiumalignment/code")
import stanly
from sklearn.cluster import KMeans

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
kNN = 24
# 1. generate a list of all coordinates 

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
nnControl = sp_spatial.distance.cdist(experimentalResults[4]['maskedTissuePositionList'], experimentalResults[4]['maskedTissuePositionList'],metric='euclidean')
nnControlSortedDist = np.sort(nnControl, axis=1)[:,1:kNN+1]
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
nnEdgeList = np.transpose([np.repeat(np.array(range(nnControlSortedIdx.shape[0])), kNN, axis=0).transpose().astype('int32'), nnControlSortedIdx.flatten().transpose().astype('int32')])
nnEdgeList = np.unique(np.sort(nnEdgeList, axis=1), axis=0)
#############################################################################
#%% takes a long time, if already run just load the adjacency data from csv
#############################################################################
# start_time = time.time()
# adjacencyDataControl = []
# for i, j in nnEdgeList[0:10,:]:
#     I = experimentalResults[4]['filteredFeatureMatrixMaskedSorted'][:,i].todense().astype('float32')
#     J = experimentalResults[4]['filteredFeatureMatrixMaskedSorted'][:,j].todense().astype('float32')
#     cs = np.sum(np.dot(I,J.transpose())) / (np.sqrt(np.sum(np.square(I)))*np.sqrt(np.sum(np.square(J))))
#     adjacencyDataControl.append(cs)
#         # print(cs)
# print("--- %s seconds ---" % (time.time() - start_time))

# start_time = time.time()
# I @ J.transpose()
# print("--- %s seconds ---" % (time.time() - start_time))
# start_time = time.time()
# np.matmul(I,J.transpose())
# print("--- %s seconds ---" % (time.time() - start_time))

# start_time = time.time()
# np.dot(I,J.transpose())
# print("--- %s seconds ---" % (time.time() - start_time))
#%% try the same as above but using list comprehension
start_time = time.time()

def cosineSimOfSample(i,j):
    I = experimentalResults[4]['filteredFeatureMatrixMaskedSorted'][:,i].todense().astype('float32')
    J = experimentalResults[4]['filteredFeatureMatrixMaskedSorted'][:,j].todense().astype('float32')
    cs = np.sum(np.dot(I,J.transpose())) / (np.sqrt(np.sum(np.square(I)))*np.sqrt(np.sum(np.square(J))))
    return cs

adjacencyDataControl = [cosineSimOfSample(i, j) for i,j in nnEdgeList]
print("--- %s seconds ---" % (time.time() - start_time))  

with open(os.path.join(derivatives,'adjacencyDataForControlSingleSample.csv'), 'w', encoding='UTF8') as f:
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
with open(os.path.join(derivatives,"adjacencyDataForControlSingleSample.csv"), newline='') as csvfile:
    tsvreader = csv.reader(csvfile, delimiter=',')
    for row in tsvreader:
        adjacencyDataControl = row
#%%
W = np.zeros([experimentalResults[4]['maskedTissuePositionList'].shape[0],experimentalResults[4]['maskedTissuePositionList'].shape[0]])
for idx, actCS in enumerate(adjacencyDataControl):
    W[nnEdgeList[idx,0],nnEdgeList[idx,1]] = 1 - float(actCS)
    # W[nnEdgeList[idx,1],nnEdgeList[idx,0]] = 1 - float(actCS)

W = (W - W.min())/(W.max() - W.min())
W[W==1] = 0
D = np.diag(sum(W))
L = D - W
# L = sp_sparse.coo_matrix(L)
clusterK = 100
eigval,eigvec = sp_sparse.linalg.eigs(L, k=30, sigma=1e-6)
eigvalSort = np.sort(np.real(eigval))
eigvalSortIdx = np.argsort(np.real(eigval))
eigvecSort = np.real(eigvec[:,eigvalSortIdx])
# evec = real(evec(:, idx))./sqrt(eval');

clusters = KMeans(n_clusters=clusterK, init='random').fit(np.real(eigvecSort))
plt.scatter(experimentalResults[4]['maskedTissuePositionList'][:,0], experimentalResults[4]['maskedTissuePositionList'][:,1],c=clusters.labels_)


