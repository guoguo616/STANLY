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
sys.path.insert(0, "/home/zjpeters/Documents/stanly/code")
import stanly
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm
# colormap of colorblind friendly colors from https://gist.github.com/thriveth/8560036
CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']

# convert to rgb 
cbRGB = []
for h in CB_color_cycle:
    hexNum = h.lstrip('#')
    cbRGB.append(tuple(int(hexNum[i:i+2], 16) for i in (0, 2, 4)))
    
cbRGB = np.array(cbRGB)/255

cbCmap = mcolors.LinearSegmentedColormap.from_list('colorblindColormap', cbRGB)


rawdata, derivatives = stanly.setExperimentalFolder("/home/zjpeters/Documents/stanly")
template = stanly.chooseTemplateSlice(70)
#%% load experiment of samples that have already been processed and registered

sampleList = []
templateList = []
with open(os.path.join(rawdata,"participants.tsv"), newline='') as tsvfile:
    tsvreader = csv.reader(tsvfile, delimiter='\t')
    next(tsvreader)
    for row in tsvreader:
        sampleList.append(row[0])
        templateList.append(row[1:])

sampleList = np.array(sampleList)
templateList = np.array(templateList, dtype='int')
# list of samples to include
imageList = [0,1,2,3,4,5,6,7,10,11,12,13,15]

experiment = {'sample-id': np.asarray(sampleList)[imageList],
                    'template-slice': templateList[imageList,0],
                    'rotation': templateList[imageList,1],
                    'experimental-group': templateList[imageList,2],
                    'flip': templateList[imageList,3]}


processedSamples = {}
totalSpotCount = 0
for actSample in range(len(experiment['sample-id'])):
    sampleData = stanly.importVisiumData(os.path.join(rawdata, experiment['sample-id'][actSample]))
    flipBool=False
    if experiment['flip'][actSample]==1:
        flipBool=True
    sampleProcessed = stanly.processVisiumData(sampleData, template, experiment['rotation'][actSample], derivatives, flip=flipBool)
    processedSamples[actSample] = sampleProcessed
    totalSpotCount += sampleProcessed['spotCount']
nTotalSamples = len(processedSamples)
spotCountMean = totalSpotCount / nTotalSamples
print(f"Average spot count across {nTotalSamples} samples is {spotCountMean}")

bestSampleToTemplate = stanly.runANTsToAllenRegistration(processedSamples[4], template, hemisphere='rightHem')

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

#%% calculate list of fully connected edges for single sample
fullyConnectedEdges = []
sampleToCluster = processedSamples[6]
for i in range(sampleToCluster['geneMatrixLog2'].shape[1]):
    for j in range(sampleToCluster['geneMatrixLog2'].shape[1]):
        fullyConnectedEdges.append([i,j])
        
fullyConnectedEdges = np.array(fullyConnectedEdges,dtype='int32')
fullyConnectedEdges = np.unique(np.sort(fullyConnectedEdges, axis=1),axis=0)

# calculate cosine sim for single sample

start_time = time.time()
sampleToClusterFilteredFeatureMatrix = np.array(sampleToCluster['geneMatrixLog2'].todense(),dtype='float32')
adjacencyDataControl = [stanly.cosineSimOfConnection(sampleToClusterFilteredFeatureMatrix,I, J) for I,J in fullyConnectedEdges]
print("--- %s seconds ---" % (time.time() - start_time))  

with open(os.path.join(derivatives,f'adjacencyDataFor{sampleToCluster["sampleID"]}DigitalSpots.csv'), 'w', encoding='UTF8') as f:
    writer = csv.writer(f)
    writer.writerow(adjacencyDataControl) 
    
# create laplacian for single sample
WsampleToCluster= np.zeros([sampleToCluster['processedTissuePositionList'].shape[0],sampleToCluster['processedTissuePositionList'].shape[0]],dtype='float32')
for idx, actCS in enumerate(adjacencyDataControl):
    WsampleToCluster[fullyConnectedEdges[idx,0],fullyConnectedEdges[idx,1]] = float(actCS)
    WsampleToCluster[fullyConnectedEdges[idx,1],fullyConnectedEdges[idx,0]] = float(actCS)
# W = sp_sparse.coo_matrix((np.array(adjacencyDataControl), (nnEdgeList[:,0],nnEdgeList[:,1])), shape=(nnControlSortedIdx.shape[0],nnControlSortedIdx.shape[0]), dtype='float32')
# W = W.todense()
WsampleToCluster = (WsampleToCluster - WsampleToCluster.min())/(WsampleToCluster.max() - WsampleToCluster.min())
WsampleToCluster[WsampleToCluster==1] = 0
DsampleToCluster = np.diag(sum(WsampleToCluster))
LsampleToCluster = DsampleToCluster - WsampleToCluster
eigvalsampleToCluster,eigvecsampleToCluster = np.linalg.eig(LsampleToCluster)
eigvalsampleToClusterSort = np.sort(np.real(eigvalsampleToCluster))[::-1]
eigvalsampleToClusterSortIdx = np.argsort(np.real(eigvalsampleToCluster))[::-1]
eigvecsampleToClusterSort = np.real(eigvecsampleToCluster[:,eigvalsampleToClusterSortIdx])

#%% run k means and silhouette analysis for single sample

clusterRange = np.array(range(4,26))

for actK in clusterRange:
    # Create a subplot with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlim([-1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, sampleToCluster['geneMatrixLog2'].shape[1] + (actK + 1) * 10])

    clusters = KMeans(n_clusters=actK, init='random', n_init=500, tol=1e-10)
    cluster_labels = clusters.fit_predict(np.real(eigvecsampleToClusterSort[:,0:actK]))

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(np.real(eigvecsampleToClusterSort[:,0:actK]), cluster_labels)
    print(
        "For n_clusters =",
        actK,
        "The average silhouette_score is :",
        silhouette_avg,
    )

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(np.real(eigvecsampleToClusterSort[:,0:actK]), cluster_labels)

    y_lower = 10
    for i in range(actK):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        # color = cm.tab20b(float(i) / actK)
        color = cbCmap(float(i) / actK)
        ax1.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # 2nd Plot showing the actual clusters formed
    colors = cbCmap(cluster_labels.astype(float) / actK)
    ax2.imshow(sampleToCluster['tissueProcessed'],cmap='gray_r')
    ax2.scatter(sampleToCluster['processedTissuePositionList'][:,0], sampleToCluster['processedTissuePositionList'][:,1],c=colors,cmap=cbCmap)
    ax2.set_title("The visualization of the clustered data.")
    ax2.axis('off')

    plt.suptitle(
        f"Silhouette analysis for KMeans clustering on {sampleToCluster['sampleID']} data with n_clusters = %d"
        % actK,
        fontsize=14,
        fontweight="bold",
    )
    plt.savefig(os.path.join(derivatives,f'clusteringAndSilhouetteSleepDep{sampleToCluster["sampleID"]}K{actK}.png'), bbox_inches='tight', dpi=300)
    plt.show()

#%% calculate a mean filtered feature matrix for control subjects
digitalControlGeneMatrix = np.zeros([nGenesInList,nDigitalSpots],dtype='float32')
nControls = 0
for actSpotIdx in range(nDigitalSpots):
    digitalControlColumn = np.zeros([nGenesInList,1],dtype='float32')
    nSpotsTotal=0
    for actSample in range(len(allSamplesToAllen)):
        if experiment['experimental-group'][actSample] == 0:
            nControls += 1
            spots = allSamplesToAllen[actSample]['digitalSpotNearestNeighbors'][actSpotIdx,:]
            if np.all(spots > 0):
                digitalControlColumn = digitalControlColumn + np.sum(allSamplesToAllen[actSample]['geneMatrixMaskedSorted'][:,spots].todense().astype('float32'), axis=1)
                nSpotsTotal+=kSpots
            
    digitalControlGeneMatrix[:,actSpotIdx] = np.array(np.divide(digitalControlColumn, nSpotsTotal),dtype='float32').flatten()

#%% calculate a mean filtered feature matrix for Experimental subjects
digitalExperimentalGeneMatrix = np.zeros([nGenesInList,nDigitalSpots],dtype='float32')
nExperimentals = 0
for actSpotIdx in range(nDigitalSpots):
    digitalExperimentalColumn = np.zeros([nGenesInList,1],dtype='float32')
    nSpotsTotal=0
    for actSample in range(len(allSamplesToAllen)):
        if experiment['experimental-group'][actSample] == 1:
            nExperimentals += 1
            spots = allSamplesToAllen[actSample]['digitalSpotNearestNeighbors'][actSpotIdx,:]
            if np.all(spots > 0):
                digitalExperimentalColumn = digitalExperimentalColumn + np.sum(allSamplesToAllen[actSample]['geneMatrixMaskedSorted'][:,spots].todense().astype('float32'), axis=1)
                nSpotsTotal+=kSpots
            
    digitalExperimentalGeneMatrix[:,actSpotIdx] = np.array(np.divide(digitalExperimentalColumn, nSpotsTotal),dtype='float32').flatten()


#%% calculate fully connected cosine sim for mean filtered feature matrix
fullyConnectedEdges = []
for i in range(nDigitalSpots):
    for j in range(nDigitalSpots):
        fullyConnectedEdges.append([i,j])
        
fullyConnectedEdges = np.array(fullyConnectedEdges,dtype='int32')
fullyConnectedEdges = np.unique(np.sort(fullyConnectedEdges, axis=1),axis=0)

#%% calculate cosine sim

start_time = time.time()
adjacencyDataControl = [stanly.cosineSimOfConnection(digitalControlFilterFeatureMatrix,i, j) for i,j in fullyConnectedEdges]
print("--- %s seconds ---" % (time.time() - start_time))  

with open(os.path.join(derivatives,'adjacencyDataForControlDigitalSpots.csv'), 'w', encoding='UTF8') as f:
    writer = csv.writer(f)
    # for i in np.array(adjacencyDataControl):
    writer.writerow(adjacencyDataControl) 
    
#%% cosine sim of experimental group
start_time = time.time()
adjacencyDataExperimental = [stanly.cosineSimOfConnection(digitalExperimentalFilterFeatureMatrix,i, j) for i,j in fullyConnectedEdges]
print("--- %s seconds ---" % (time.time() - start_time))  

with open(os.path.join(derivatives,'adjacencyDataForExperimentalDigitalSpots.csv'), 'w', encoding='UTF8') as f:
    writer = csv.writer(f)
    # for i in np.array(adjacencyDataExperimental):
    writer.writerow(adjacencyDataExperimental) 
    
    
#%% create laplacian for control
Wcontrol= np.zeros([nDigitalSpots,nDigitalSpots],dtype='float32')
for idx, actCS in enumerate(adjacencyDataControl):
    Wcontrol[fullyConnectedEdges[idx,0],fullyConnectedEdges[idx,1]] = float(actCS)
    Wcontrol[fullyConnectedEdges[idx,1],fullyConnectedEdges[idx,0]] = float(actCS)
# W = sp_sparse.coo_matrix((np.array(adjacencyDataControl), (nnEdgeList[:,0],nnEdgeList[:,1])), shape=(nnControlSortedIdx.shape[0],nnControlSortedIdx.shape[0]), dtype='float32')
# W = W.todense()
Wcontrol = (Wcontrol - Wcontrol.min())/(Wcontrol.max() - Wcontrol.min())
Wcontrol[Wcontrol==1] = 0
Dcontrol = np.diag(sum(Wcontrol))
Lcontrol = Dcontrol - Wcontrol
eigvalControl,eigvecControl = np.linalg.eig(Lcontrol)
eigvalControlSort = np.sort(np.real(eigvalControl))[::-1]
eigvalControlSortIdx = np.argsort(np.real(eigvalControl))[::-1]
eigvecControlSort = np.real(eigvecControl[:,eigvalControlSortIdx])

#%% run k means control
clusterK = 15
clusters = KMeans(n_clusters=clusterK, init='random', n_init=300, tol=1e-8,).fit(np.real(eigvecControlSort[:,0:clusterK]))
plt.imshow(allSamplesToAllen[4]['tissueRegistered'],cmap='gray')
plt.scatter(templateDigitalSpots[:,0], templateDigitalSpots[:,1],c=clusters.labels_,cmap='Set2')

#%% create laplacian for experimental
Wexperimental= np.zeros([nDigitalSpots,nDigitalSpots],dtype='float32')
for idx, actCS in enumerate(adjacencyDataExperimental):
    Wexperimental[fullyConnectedEdges[idx,0],fullyConnectedEdges[idx,1]] = float(actCS)
    Wexperimental[fullyConnectedEdges[idx,1],fullyConnectedEdges[idx,0]] = float(actCS)
# W = sp_sparse.coo_matrix((np.array(adjacencyDataExperimental), (nnEdgeList[:,0],nnEdgeList[:,1])), shape=(nnExperimentalSortedIdx.shape[0],nnExperimentalSortedIdx.shape[0]), dtype='float32')
# W = W.todense()
Wexperimental = (Wexperimental - Wexperimental.min())/(Wexperimental.max() - Wexperimental.min())
Wexperimental[Wexperimental==1] = 0
Dexperimental = np.diag(sum(Wexperimental))
Lexperimental = Dexperimental - Wexperimental
eigvalExperimental,eigvecExperimental = np.linalg.eig(Lexperimental)
eigvalExperimentalSort = np.sort(np.real(eigvalExperimental))[::-1]
eigvalExperimentalSortIdx = np.argsort(np.real(eigvalExperimental))[::-1]
eigvecExperimentalSort = np.real(eigvecExperimental[:,eigvalExperimentalSortIdx])

#%% run k means experimental
clusterK = 15
clusters = KMeans(n_clusters=clusterK, init='random', n_init=300, tol=1e-8,).fit(np.real(eigvecExperimentalSort[:,0:clusterK]))
plt.imshow(allSamplesToAllen[4]['tissueRegistered'],cmap='gray')
plt.scatter(templateDigitalSpots[:,0], templateDigitalSpots[:,1],c=clusters.labels_,cmap='tab20b')

#%% run sillhoutte analysis on control clustering

clusterRange = np.array(range(4,26))

for actK in clusterRange:
    # Create a subplot with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlim([-1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, nDigitalSpots + (actK + 1) * 10])

    clusters = KMeans(n_clusters=actK, init='random', n_init=500, tol=1e-8,)
    cluster_labels = clusters.fit_predict(np.real(eigvecControlSort[:,0:actK]))

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(np.real(eigvecControlSort[:,0:actK]), cluster_labels)
    print(
        "For n_clusters =",
        actK,
        "The average silhouette_score is :",
        silhouette_avg,
    )

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(np.real(eigvecControlSort[:,0:actK]), cluster_labels)

    y_lower = 10
    for i in range(actK):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.tab20b(float(i) / actK)
        ax1.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # 2nd Plot showing the actual clusters formed
    colors = cm.tab20b(cluster_labels.astype(float) / actK)
    ax2.imshow(allSamplesToAllen[4]['tissueRegistered'],cmap='gray_r')
    ax2.scatter(templateDigitalSpots[:,0], templateDigitalSpots[:,1],c=colors,cmap='tab20b')
    ax2.set_title("The visualization of the clustered control data.")
    ax2.axis('off')

    plt.suptitle(
        "Silhouette analysis for KMeans clustering on control sample data with n_clusters = %d"
        % actK,
        fontsize=14,
        fontweight="bold",
    )
    plt.savefig(os.path.join(derivatives,f'clusteringAndSilhouetteSleepDepControlK{actK}.png'), bbox_inches='tight', dpi=300)
    plt.show()
    
#%% run sillhoutte analysis on Experimental clustering
clusterRange = np.array(range(4,26))

for actK in clusterRange:
    # Create a subplot with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlim([-1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, nDigitalSpots + (actK + 1) * 10])

    clusters = KMeans(n_clusters=actK, init='random', n_init=300, tol=1e-8,)
    cluster_labels = clusters.fit_predict(np.real(eigvecExperimentalSort[:,0:actK]))

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(np.real(eigvecExperimentalSort[:,0:actK]), cluster_labels)
    print(
        "For n_clusters =",
        actK,
        "The average silhouette_score is :",
        silhouette_avg,
    )

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(np.real(eigvecExperimentalSort[:,0:actK]), cluster_labels)

    y_lower = 10
    for i in range(actK):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.tab20b(float(i) / actK)
        ax1.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # 2nd Plot showing the actual clusters formed
    colors = cm.tab20b(cluster_labels.astype(float) / actK)
    ax2.imshow(allSamplesToAllen[4]['tissueRegistered'],cmap='gray_r')
    ax2.scatter(templateDigitalSpots[:,0], templateDigitalSpots[:,1],c=colors,cmap='tab20b')
    ax2.set_title("The visualization of the clustered Experimental data.")
    ax2.axis('off')

    plt.suptitle(
        "Silhouette analysis for KMeans clustering on Experimental sample data with n_clusters = %d"
        % actK,
        fontsize=14,
        fontweight="bold",
    )
    plt.savefig(os.path.join(derivatives,f'clusteringAndSilhouetteSleepDepExperimentalK{actK}.png'), bbox_inches='tight', dpi=300)
    plt.show()

#%% preparing pca

from sklearn.decomposition import PCA
components=20
pca = PCA(n_components=components)
controlPca = pca.fit(digitalControlFilterFeatureMatrix.transpose())

expVar = controlPca.explained_variance_ratio_
cumExpVar = np.cumsum(controlPca.explained_variance_ratio_)
# Plot the explained variance
x = ["PC%s" %i for i in range(1,components+1)]
expVarBar = plt.bar(x, expVar)
cumExpVarLine = plt.scatter(x,cumExpVar)
plt.show()

controlPcaFeatures = pca.fit_transform(digitalControlFilterFeatureMatrix.transpose())

clusterK = 25
clusters = KMeans(n_clusters=clusterK, init='random', n_init=300, tol=1e-8,).fit(controlPcaFeatures)
plt.imshow(allSamplesToAllen[4]['tissueRegistered'],cmap='gray')
plt.scatter(templateDigitalSpots[:,0], templateDigitalSpots[:,1],c=clusters.labels_,cmap='Set2')



#%% pca with z-score normalized spots 
# WORKS BETTER WITHOUT Z-SCORE
# z = (x-u) / std
# digitalControlFilterFeatureMatrixMean = np.mean(digitalControlFilterFeatureMatrix, axis=1)
# digitalControlFilterFeatureMatrixStd = np.std(digitalControlFilterFeatureMatrix, axis=1)
# digitalControlFilterFeatureMatrixZ = digitalControlFilterFeatureMatrix - digitalControlFilterFeatureMatrixMean[:,np.newaxis]
# digitalControlFilterFeatureMatrixZ = np.divide(digitalControlFilterFeatureMatrixZ, digitalControlFilterFeatureMatrixStd[:,np.newaxis])

# components=20
# pca = PCA(n_components=components)
# controlPca = pca.fit(digitalControlFilterFeatureMatrixZ.transpose())

# expVar = controlPca.explained_variance_ratio_
# cumExpVar = np.cumsum(controlPca.explained_variance_ratio_)
# # Plot the explained variance
# x = ["PC%s" %i for i in range(1,components+1)]
# expVarBar = plt.bar(x, expVar)
# cumExpVarLine = plt.scatter(x,cumExpVar)
# plt.show()

# controlPcaFeatures = pca.fit_transform(digitalControlFilterFeatureMatrixZ.transpose())

# clusters = KMeans(n_clusters=clusterK, init='random', n_init=300, tol=1e-8,).fit(controlPcaFeatures)
# plt.imshow(allSamplesToAllen[4]['tissueRegistered'],cmap='gray')
# plt.scatter(templateDigitalSpots[:,0], templateDigitalSpots[:,1],c=clusters.labels_,cmap='Set2')
#%% run analysis for the delta of control and Delta digital filtered feature matrix
#%% cosine sim of Delta group
start_time = time.time()
digitalDeltaFilterFeatureMatrix = digitalControlFilterFeatureMatrix - digitalExperimentalFilterFeatureMatrix
adjacencyDataDelta = [stanly.cosineSimOfConnection(digitalDeltaFilterFeatureMatrix,i, j) for i,j in fullyConnectedEdges]
print("--- %s seconds ---" % (time.time() - start_time))  

with open(os.path.join(derivatives,'adjacencyDataForDeltaDigitalSpots.csv'), 'w', encoding='UTF8') as f:
    writer = csv.writer(f)
    # for i in np.array(adjacencyDataDelta):
    writer.writerow(adjacencyDataDelta) 

#%% create laplacian for Delta
WDelta= np.zeros([nDigitalSpots,nDigitalSpots],dtype='float32')
for idx, actCS in enumerate(adjacencyDataDelta):
    WDelta[fullyConnectedEdges[idx,0],fullyConnectedEdges[idx,1]] = float(actCS)
    WDelta[fullyConnectedEdges[idx,1],fullyConnectedEdges[idx,0]] = float(actCS)
# W = sp_sparse.coo_matrix((np.array(adjacencyDataDelta), (nnEdgeList[:,0],nnEdgeList[:,1])), shape=(nnDeltaSortedIdx.shape[0],nnDeltaSortedIdx.shape[0]), dtype='float32')
# W = W.todense()
WDelta = (WDelta - WDelta.min())/(WDelta.max() - WDelta.min())
WDelta[WDelta==1] = 0
DDelta = np.diag(sum(WDelta))
LDelta = DDelta - WDelta
eigvalDelta,eigvecDelta = np.linalg.eig(LDelta)
eigvalDeltaSort = np.sort(np.real(eigvalDelta))[::-1]
eigvalDeltaSortIdx = np.argsort(np.real(eigvalDelta))[::-1]
eigvecDeltaSort = np.real(eigvecDelta[:,eigvalDeltaSortIdx])

#%% run sillhoutte analysis on Delta clustering
clusterRange = np.array(range(3,26))

for actK in clusterRange:
    # Create a subplot with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlim([-1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, nDigitalSpots + (actK + 1) * 10])

    clusters = KMeans(n_clusters=actK, init='random', n_init=300, tol=1e-8,)
    cluster_labels = clusters.fit_predict(np.real(eigvecDeltaSort[:,0:actK]))

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(np.real(eigvecDeltaSort[:,0:actK]), cluster_labels)
    print(
        "For n_clusters =",
        actK,
        "The average silhouette_score is :",
        silhouette_avg,
    )

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(np.real(eigvecDeltaSort[:,0:actK]), cluster_labels)

    y_lower = 10
    for i in range(actK):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.tab20b(float(i) / actK)
        ax1.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # 2nd Plot showing the actual clusters formed
    colors = cm.tab20b(cluster_labels.astype(float) / actK)
    ax2.imshow(allSamplesToAllen[4]['tissueRegistered'],cmap='gray_r')
    ax2.scatter(templateDigitalSpots[:,0], templateDigitalSpots[:,1],c=colors,cmap='tab20b')
    ax2.set_title("The visualization of the clustered Delta data.")
    ax2.axis('off')

    plt.suptitle(
        "Silhouette analysis for KMeans clustering on Delta sample data with n_clusters = %d"
        % actK,
        fontsize=14,
        fontweight="bold",
    )
    plt.savefig(os.path.join(derivatives,f'clusteringAndSilhouetteSleepDepDeltaK{actK}.png'), bbox_inches='tight', dpi=300)
    plt.show()

