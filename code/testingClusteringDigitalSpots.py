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


#%% create digital spots for whole slice and find nearest neighbors
# ONLY RUN ONE OF THE FOLLOWING TWO SECTIONS, OTHERWISE
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

nDigitalSpots = len(templateDigitalSpots)
nSampleExperimental = sum(experiment['experimental-group'])
nSampleControl = len(experiment['experimental-group']) - nSampleExperimental
nGenesInList = len(allSampleGeneList)

for sampleIdx, actSample in enumerate(allSamplesToAllen):
    allSamplesToAllen[sampleIdx]['allSampleGeneList'] = allSampleGeneList 
    sortedIdxList = np.zeros(nGenesInList,dtype=int)
    for sortedIdx, actGene in enumerate(allSampleGeneList):
        sortedIdxList[sortedIdx] = allSamplesToAllen[sampleIdx]['geneListMasked'].index(actGene)
    allSamplesToAllen[sampleIdx]['filteredFeatureMatrixMaskedSorted'] = allSamplesToAllen[sampleIdx]['filteredFeatureMatrixMasked'][sortedIdxList,:].astype('int32')
    allSamplesToAllen[sampleIdx].pop('filteredFeatureMatrixMasked')
    allSamplesToAllen[sampleIdx].pop('geneListMasked')




#%% calculate a mean filtered feature matrix for control subjects
digitalControlFilterFeatureMatrix = np.zeros([nGenesInList,nDigitalSpots],dtype='float32')
nControls = 0
# nanSpots = np.zeros([nGenesInList,kSpots],dtype='float32')
# nanSpots[nanSpots==0] = np.nan
for actSpotIdx in range(nDigitalSpots):
    digitalControlColumn = np.zeros([nGenesInList,1],dtype='float32')
    nSpotsTotal=0
    for actSample in range(len(allSamplesToAllen)):
        if experiment['experimental-group'][actSample] == 0:
            nControls += 1
    # geneIndex = allSamplesToAllen[actSample]['geneListMasked'].index(actGene)
            spots = allSamplesToAllen[actSample]['digitalSpotNearestNeighbors'][actSpotIdx,:]
            if np.all(spots > 0):
                digitalControlColumn = digitalControlColumn + np.sum(allSamplesToAllen[actSample]['filteredFeatureMatrixMaskedSorted'][:,spots].todense().astype('float32'), axis=1)
                nSpotsTotal+=kSpots
            
    digitalControlFilterFeatureMatrix[:,actSpotIdx] = np.array(np.divide(digitalControlColumn, nSpotsTotal),dtype='float32').flatten()

#%% calculate fully connected cosine sim for mean filtered feature matrix
fullyConnectedEdges = []
for i in range(nDigitalSpots):
    for j in range(nDigitalSpots):
        fullyConnectedEdges.append([i,j])
        
fullyConnectedEdges = np.array(fullyConnectedEdges)
fullyConnectedEdges = np.unique(np.sort(fullyConnectedEdges, axis=1),axis=0)

#%% cosine sim
start_time = time.time()

def cosineSimOfConnection(i,j):
    I = digitalControlFilterFeatureMatrix[:,i]
    J = digitalControlFilterFeatureMatrix[:,j]
    # cs = np.sum(np.dot(I,J.transpose())) / (np.sqrt(np.sum(np.square(I)))*np.sqrt(np.sum(np.square(J))))
    cs = sp_spatial.distance.cosine(I,J)
    return cs

adjacencyDataControl = [cosineSimOfConnection(i, j) for i,j in fullyConnectedEdges]
print("--- %s seconds ---" % (time.time() - start_time))  

with open(os.path.join(derivatives,'adjacencyDataForControlDigitalSpots.csv'), 'w', encoding='UTF8') as f:
    writer = csv.writer(f)
    # for i in np.array(adjacencyDataControl):
    writer.writerow(adjacencyDataControl) 
    
#%% create laplacian 
W = np.zeros([nDigitalSpots,nDigitalSpots],dtype='float32')
for idx, actCS in enumerate(adjacencyDataControl):
    W[fullyConnectedEdges[idx,0],fullyConnectedEdges[idx,1]] = float(actCS)
    W[fullyConnectedEdges[idx,1],fullyConnectedEdges[idx,0]] = float(actCS)
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

#%% run k means
clusterK = 10
clusters = KMeans(n_clusters=clusterK, init='random', n_init=300, tol=1e-8,).fit(np.real(eigvecSort[:,0:clusterK]))
plt.imshow(allSamplesToAllen[4]['tissueRegistered'],cmap='gray')
plt.scatter(templateDigitalSpots[:,0], templateDigitalSpots[:,1],c=clusters.labels_,cmap='Set2')

#%% show overlay of images with template

plt.imshow(template['rightHem'])
plt.imshow(allSamplesToAllen[4]['tissueRegistered'], cmap='gray',alpha=0.6)
plt.axis('off')
plt.show()
#%% show overlay of images with template

plt.imshow(allSamplesToAllen[4]['visiumTransformed'])
plt.imshow(allSamplesToAllen[5]['visiumTransformed'], cmap='gray',alpha=0.6)
plt.axis('off')
plt.show()
