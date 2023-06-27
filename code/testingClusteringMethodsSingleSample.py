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
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering

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
    sample = stanly.importVisiumData(os.path.join(rawdata, experiment['sample-id'][actSample]))
    sampleProcessed = stanly.processVisiumData(sample, template, experiment['rotation'][actSample], derivatives)
    # sampleProcessed = stanly.loadProcessedSample(os.path.join(derivatives, experiment['sample-id'][actSample]))
    processedSamples[actSample] = sampleProcessed
    totalSpotCount += sampleProcessed['spotCount']
nTotalSamples = len(processedSamples)
spotCountMean = totalSpotCount / nTotalSamples
print(f"Average spot count across {nTotalSamples} samples is {spotCountMean}")

bestSample = processedSamples[4]

bestSampleToTemplate = stanly.runANTsToAllenRegistration(bestSample, template)

experimentalResults = {}
for actSample in range(len(processedSamples)):
    sampleRegistered = stanly.runANTsInterSampleRegistration(processedSamples[actSample], bestSample)
    experimentalResults[actSample] = sampleRegistered

# allSamplesToAllen = {}
for actSample in range(len(experimentalResults)):
    regSampleToTemplate = stanly.applyAntsTransformations(experimentalResults[actSample], bestSampleToTemplate, template)
    experimentalResults[actSample] = regSampleToTemplate

# experimentalResults = {}
# for actSample in range(len(experiment['sample-id'])):
#     sampleRegistered = stanly.loadAllenRegisteredSample(os.path.join(derivatives, experiment['sample-id'][actSample]))
#     experimentalResults[actSample] = sampleRegistered
#     experimentalResults[actSample]['experimentalStatus'] = experiment['experimental-group'][actSample]


    #%% create list of genes expressed in all samples
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
    experimentalResults[sampleIdx]['filteredFeatureMatrixMaskedSorted'] = experimentalResults[sampleIdx]['filteredFeatureMatrixMasked'][sortedIdxList,:].astype('int32')
    experimentalResults[sampleIdx].pop('filteredFeatureMatrixMasked')
    experimentalResults[sampleIdx].pop('geneListMasked')
        # experimentalResults[i]['geneListSortedForGroup'][sortedIdx]

#%% perform spectral clustering on groups using locations in registered coordinates
# could consider breaking control and experimental into separate sections to save on memory
sampleToCluster=4
nControls = 0
nExperimentals = 0
kNN = 48 #experimentalResults[sampleToCluster]['maskedTissuePositionList'].shape[0]
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
nnControl = sp_spatial.distance.cdist(experimentalResults[sampleToCluster]['maskedTissuePositionList'], experimentalResults[sampleToCluster]['maskedTissuePositionList'],metric='sqeuclidean').astype('float32')
# nnControlSortedDist = np.sort(nnControl, axis=1)[:,1:kNN+1]
# nnControlSortedIdx = np.argsort(nnControl, axis=1)[:,1:kNN+1]
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

#%% testing thresholding
sampleStd = np.std(experimentalResults[sampleToCluster]['filteredFeatureMatrixMaskedSorted'].todense().astype('float32'), axis=1)
sampleStdSort = np.argsort(sampleStd, axis=0)
for geneIdx in sampleStdSort[6000:7000]:
    plt.imshow(experimentalResults[sampleToCluster]['tissueRegistered'],cmap='gray')
    plt.scatter(experimentalResults[sampleToCluster]['maskedTissuePositionList'][:,0], experimentalResults[sampleToCluster]['maskedTissuePositionList'][:,1],c=np.array(experimentalResults[sampleToCluster]['filteredFeatureMatrixMaskedSorted'].todense()[geneIdx,:]),cmap='Reds',alpha=0.8)
    plt.show()

#%% remove duplicates from nearest neighbor list
# nnEdgeList = np.transpose([np.repeat(np.array(range(nnControlSortedIdx.shape[0])), kNN, axis=0).astype('int32'), nnControlSortedIdx.flatten().astype('int32')])
# nnEdgeList = np.unique(np.sort(nnEdgeList, axis=1), axis=0)
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

def cosineSimOfConnection(i,j):
    I = experimentalResults[sampleToCluster]['filteredFeatureMatrixMaskedSorted'][:,i].todense().astype('float32')
    J = experimentalResults[sampleToCluster]['filteredFeatureMatrixMaskedSorted'][:,j].todense().astype('float32')
    # cs = np.sum(np.dot(I,J.transpose())) / (np.sqrt(np.sum(np.square(I)))*np.sqrt(np.sum(np.square(J))))
    cs = sp_spatial.distance.cosine(I,J)
    return cs

adjacencyDataControl = [cosineSimOfConnection(i, j) for i,j in nnEdgeList]
print("--- %s seconds ---" % (time.time() - start_time))  

with open(os.path.join(derivatives,f'adjacencyDataForControlSingleSample_Sample-{experimentalResults[sampleToCluster]["sampleID"]}.csv'), 'w', encoding='UTF8') as f:
    writer = csv.writer(f)
    # for i in np.array(adjacencyDataControl):
    writer.writerow(adjacencyDataControl)          
    # spotDiagMatrixControl = np.diag(sum(spotAdjacencyMatrixControl))
    # spotDiagMatrixExperimental = np.diag(sum(spotAjacencyMatrixExperimental))
    # laplacianControl = spotDiagMatrixControl - spotAdjacencyMatrixControl
    # laplacianExperimental = spotDiagMatrixExperimental -spotAdjacencyMatrixExperimental
    
    
#%%
# adjacencyDataControl = []
# # with open(os.path.join(derivatives,f'adjacencyDataForControl.csv'), 'w', encoding='UTF8') as f:
# with open(os.path.join(derivatives,"adjacencyDataForControlSingleSample.csv"), newline='') as csvfile:
#     tsvreader = csv.reader(csvfile, delimiter=',')
#     for row in tsvreader:
#         adjacencyDataControl = row
#%%
W = np.zeros([experimentalResults[sampleToCluster]['maskedTissuePositionList'].shape[0],experimentalResults[sampleToCluster]['maskedTissuePositionList'].shape[0]])
for idx, actCS in enumerate(adjacencyDataControl):
    W[nnEdgeList[idx,0],nnEdgeList[idx,1]] = float(actCS)
    W[nnEdgeList[idx,1],nnEdgeList[idx,0]] = float(actCS)
# W = sp_sparse.coo_matrix((np.array(adjacencyDataControl), (nnEdgeList[:,0],nnEdgeList[:,1])), shape=(nnControlSortedIdx.shape[0],nnControlSortedIdx.shape[0]), dtype='float32')
# W = W.todense()
W = (W - W.min())/(W.max() - W.min())
W[W==1] = 0
D = np.diag(sum(W))
L = D - W
eigval,eigvec = np.linalg.eig(L)
eigvalSort = np.sort(np.real(eigval))[::-1]
eigvalSortIdx = np.argsort(np.real(eigval))[::-1]
eigvecSort = np.real(eigvec[:,eigvalSortIdx])

#%% test image segmentation of histology
from skimage import filters
import ants
image = ants.from_numpy(experimentalResults[sampleToCluster]['tissueRegistered'])
imageMask = np.array(experimentalResults[sampleToCluster]['tissueRegistered'] > 0, dtype='uint8')
imageMask = ants.from_numpy(imageMask)
ants.atropos(image, imageMask)
#%% run k means
clusterK = 50
clusters = KMeans(n_clusters=clusterK, init='random', n_init=300, tol=1e-8,).fit(np.real(eigvecSort[:,0:clusterK]))
plt.imshow(experimentalResults[sampleToCluster]['tissueRegistered'],cmap='gray')
plt.scatter(experimentalResults[sampleToCluster]['maskedTissuePositionList'][:,0], experimentalResults[sampleToCluster]['maskedTissuePositionList'][:,1],c=clusters.labels_,cmap='tab20c')

#%% test spectral clustering
clusterK = 15
# clusters = KMeans(n_clusters=clusterK, init='random', n_init=300, tol=1e-8,).fit(np.real(eigvecSort[:,0:clusterK]))
sc = SpectralClustering(clusterK, affinity='precomputed', n_init=300, assign_labels='discretize', n_neighbors=10)
clusters = sc.fit_predict(W)
plt.imshow(experimentalResults[sampleToCluster]['tissueRegistered'],cmap='gray')
plt.scatter(experimentalResults[sampleToCluster]['maskedTissuePositionList'][:,0], experimentalResults[sampleToCluster]['maskedTissuePositionList'][:,1],c=clusters,cmap='tab20c')

#%% test dbscan
clusterK = 50
clusters = DBSCAN(eps=0.1, min_samples=3).fit(np.real(eigvec[:,0:clusterK]))
plt.imshow(bestSampleToTemplate['tissueRegistered'],cmap='gray')
plt.scatter(experimentalResults[sampleToCluster]['maskedTissuePositionList'][:,0], experimentalResults[sampleToCluster]['maskedTissuePositionList'][:,1],c=clusters.labels_,cmap='tab20c')
#%%

# sparse approach
# L = sp_sparse.coo_matrix(L)
# eigval,eigvec = sp_sparse.linalg.eigs(L, k=300, sigma=1e-6)
# eigvalSort = np.sort(np.real(eigval))
# eigvalSortIdx = np.argsort(np.real(eigval))
# eigvecSort = np.real(eigvec[:,eigvalSortIdx])
# # evec = real(evec(:, idx))./sqrt(eval');

# clusters = KMeans(n_clusters=clusterK, init='random').fit(np.real(eigvec[:,0:clusterK]))
# plt.scatter(experimentalResults[4]['maskedTissuePositionList'][:,0], experimentalResults[4]['maskedTissuePositionList'][:,1],c=clusters.labels_)


