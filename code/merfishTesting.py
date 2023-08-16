#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 14:37:22 2023

@author: zjpeters
"""
import os
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from skimage.transform import rescale, rotate, resize
import itk
import sys
sys.path.insert(0, "/home/zjpeters/Documents/stanly/code")
import stanly
from glob import glob
from skimage import io, filters, color, feature, morphology
import csv
import cv2
from skimage.exposure import match_histograms
import scipy.sparse as sp_sparse
import json
import time
from sklearn.cluster import KMeans
import matplotlib.cm as cm
from sklearn.metrics import silhouette_samples, silhouette_score

rawdata, derivatives = stanly.setExperimentalFolder("/media/zjpeters/Samsung_T5/merscope/")
sourcedata = os.path.join(rawdata,'Slide1_Apr24')

# starting from the importVisiumData and processVisiumData function, create merfish equivalents
# expected merfish data includes:
# 1. image data, needs downsampled due to very high resolution
# 2. cell_by_gene.csv containing cell index and barcode in first two columns followed by columns of rna expression per gene per cell
# 3. cell_metadata.csv containing cell index, barcode, fov, volume, center_x, center_y, min_x, min_y, max_x, max_y

#load allen template
templateData = stanly.chooseTemplateSlice(70)

#%% import and align sample data to allen ccf

sampleData = stanly.importMerfishData(sourcedata, derivatives)
processedSample = stanly.processMerfishData(sampleData, templateData, 210, derivatives)
# sampleRegistered = stanly.runANTsToAllenRegistration(processedSample, templateData)
# plt.imshow(templateData['wholeBrain'], cmap='gray')
# plt.show()
# plt.imshow(processedSample['tissueProcessed'], cmap='gray')
# plt.scatter(processedSample['processedTissuePositionList'][:,0], processedSample['processedTissuePositionList'][:,1])
# plt.axis(False)
# plt.savefig(os.path.join(derivatives,'allen_slice_90.png'), bbox_inches='tight', dpi=300)
# plt.show()


#%% split two samples from one image using lasso tool

selectorRight = stanly.SelectUsingLasso(processedSample)
#%%
selectorRight.outputMaskedSpots()
selectorRight.outputMaskedImage(processedSample)
rightHem = selectorRight.outputMaskedSample(processedSample, 'rightHem')
# = selector.maskedSpots
#%% load experiment of samples that have already been processed and registered

# sampleList = []
# templateList = []
# with open(os.path.join(rawdata,"participants.tsv"), newline='') as tsvfile:
#     tsvreader = csv.reader(tsvfile, delimiter='\t')
#     next(tsvreader)
#     for row in tsvreader:
#         sampleList.append(row[0])
#         templateList.append(row[1:])

# sampleList = np.array(sampleList)
# templateList = np.array(templateList, dtype='int')
# # list of samples to include
# imageList = [0,1,2,3,4,5,6,7,10,11,12,13,15]

# experiment = {'sample-id': np.asarray(sampleList)[imageList],
#                     'template-slice': templateList[imageList,0],
#                     'rotation': templateList[imageList,1],
#                     'experimental-group': templateList[imageList,2],
#                     'flip': templateList[imageList,3]}



totalSpotCount = 0
# for actSample in range(len(experiment['sample-id'])):
#     sampleData = stanly.importVisiumData(os.path.join(rawdata, experiment['sample-id'][actSample]))
#     flipBool=False
#     if experiment['flip'][actSample]==1:
#         flipBool=True
#     sampleProcessed = stanly.processVisiumData(sampleData, template, experiment['rotation'][actSample], derivatives, flip=flipBool)
# processedSampleRight = {}
# processedSampleRight['degreesOfRotation']  = processedSample['degreesOfRotation']
# processedSampleRight['derivativesPath'] = f"{processedSample['derivativesPath']}_rightHem"
# processedSampleRight['geneList'] = processedSample['geneList']
# processedSampleRight['geneListMasked'] = processedSample['geneListMasked']
# processedSampleRight['geneMatrix'] = processedSample['geneMatrix'][:,selectorRight.ind]
# processedSampleRight['geneMatrixLog2'] = processedSample['geneMatrixLog2'].todense()
# processedSampleRight['geneMatrixLog2'] = processedSampleRight['geneMatrixLog2'][:,selectorRight.ind]
# processedSampleRight['processedTissuePositionList'] = selectorRight.maskedSpots
# processedSampleRight['sampleID'] = f"{processedSample['sampleID']}_rightHem"
# processedSampleRight['tissuePositionList'] = processedSample['tissuePositionList']
# processedSampleRight['tissueProcessed'] = selectorRight.maskedImage

#%%
selectorLeft = stanly.SelectUsingLasso(processedSample)

#%%
selectorLeft.outputMaskedSpots()
selectorLeft.outputMaskedImage(processedSample)
leftHem = selectorLeft.outputMaskedSample(processedSample, 'leftHem')
# processedSampleLeft = {}
# processedSampleLeft['degreesOfRotation']  = processedSample['degreesOfRotation']
# processedSampleLeft['derivativesPath'] = f"{processedSample['derivativesPath']}_LeftHem"
# processedSampleLeft['geneList'] = processedSample['geneList']
# processedSampleLeft['geneListMasked'] = processedSample['geneListMasked']
# processedSampleLeft['geneMatrix'] = processedSample['geneMatrix'][:,selectorLeft.ind]
# processedSampleLeft['geneMatrixLog2'] = processedSample['geneMatrixLog2'].todense()
# processedSampleLeft['geneMatrixLog2'] = processedSampleLeft['geneMatrixLog2'][:,selectorLeft.ind]
# processedSampleLeft['processedTissuePositionList'] = selectorLeft.maskedSpots
# processedSampleLeft['sampleID'] = f"{processedSample['sampleID']}_LeftHem"
# processedSampleLeft['tissuePositionList'] = processedSample['tissuePositionList']
# processedSampleLeft['tissueProcessed'] = selectorLeft.maskedImage

#%% register processed samples
bestSampleToTemplate = stanly.runANTsToAllenRegistration(rightHem, templateData, hemisphere='rightHem')

#%% test digital spot creation using merfish to perform clustering
wholeBrainSpotSize = 10
templateDigitalSpots = stanly.createDigitalSpots(bestSampleToTemplate, wholeBrainSpotSize)
nDigitalSpots = len(templateDigitalSpots)
nGenesInList = len(bestSampleToTemplate['geneListMasked'])
kSpots = 12
actNN, actCDist = stanly.findDigitalNearestNeighbors(templateDigitalSpots, bestSampleToTemplate['maskedTissuePositionList'], kSpots, wholeBrainSpotSize)

#%% create digital spot gene matrix 
digitalGeneMatrix = np.zeros([nGenesInList,nDigitalSpots],dtype='float32')
nControls = 0
for actSpotIdx in range(nDigitalSpots):
    # digitalGeneColumn = np.zeros([nGenesInList,1],dtype='float32')
    nSpotsTotal=0
    spots = actNN[actSpotIdx,:]
    if np.all(spots > 0):
        digitalGeneColumn = np.median(bestSampleToTemplate['geneMatrixMasked'][:,spots].todense().astype('float32'), axis=1)
        nSpotsTotal+=kSpots
        # digitalGeneMatrix[:,actSpotIdx] = np.array(np.divide(digitalGeneColumn, nSpotsTotal),dtype='float32').flatten()
        digitalGeneMatrix[:,actSpotIdx] = np.array(digitalGeneColumn,dtype='float32').flatten()
#%% calculate fully connected cosine sim for mean filtered feature matrix
fullyConnectedEdges = []
for i in range(nDigitalSpots):
    for j in range(nDigitalSpots):
        fullyConnectedEdges.append([i,j])
        
fullyConnectedEdges = np.array(fullyConnectedEdges,dtype='int32')
fullyConnectedEdges = np.unique(np.sort(fullyConnectedEdges, axis=1),axis=0)

#%% calculate cosine sim

start_time = time.time()
adjacencyData = [stanly.cosineSimOfConnection(digitalGeneMatrix,i, j) for i,j in fullyConnectedEdges]
print("--- %s seconds ---" % (time.time() - start_time))  

with open(os.path.join(derivatives,'adjacencyDataForRightHemDigitalSpots.csv'), 'w', encoding='UTF8') as f:
    writer = csv.writer(f)
    # for i in np.array(adjacencyDataControl):
    writer.writerow(adjacencyData) 
    
#%% create laplacian for control
Wcontrol= np.zeros([nDigitalSpots,nDigitalSpots],dtype='float32')
for idx, actCS in enumerate(adjacencyData):
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
    ax2.imshow(bestSampleToTemplate['tissueRegistered'],cmap='gray_r')
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


#%% test digital spot creation using merfish to perform clustering
wholeBrainSpotSize = 5
templateDigitalSpots = stanly.createDigitalSpots(sampleRegistered, wholeBrainSpotSize)
nDigitalSpots = len(templateDigitalSpots)
nGenesInList = len(sampleRegistered['geneListMasked'])
kSpots = 12
actNN, actCDist = stanly.findDigitalNearestNeighbors(templateDigitalSpots, sampleRegistered['maskedTissuePositionList'], kSpots, wholeBrainSpotSize)

#%% create digital spot gene matrix 
digitalGeneMatrix = np.zeros([nGenesInList,nDigitalSpots],dtype='float32')
nControls = 0
for actSpotIdx in range(nDigitalSpots):
    # digitalGeneColumn = np.zeros([nGenesInList,1],dtype='float32')
    nSpotsTotal=0
    spots = actNN[actSpotIdx,:]
    if np.all(spots > 0):
        digitalGeneColumn = np.median(sampleRegistered['geneMatrixMasked'][:,spots].todense().astype('float32'), axis=1)
        nSpotsTotal+=kSpots
        # digitalGeneMatrix[:,actSpotIdx] = np.array(np.divide(digitalGeneColumn, nSpotsTotal),dtype='float32').flatten()
        digitalGeneMatrix[:,actSpotIdx] = np.array(digitalGeneColumn,dtype='float32').flatten()
    # else:
    #     digitalGeneColumn = np.zeros([nGenesInList,1],dtype='float32')
    #     digitalGeneMatrix[:,actSpotIdx] = digitalGeneColumn.flatten()

#%% make 0 values nan
digitalGeneMatrixNaN = np.array(digitalGeneMatrix, dtype='double')
digitalGeneMatrixNaN[digitalGeneMatrixNaN == 0] = np.nan
for geneIdx,actGene in enumerate(sampleRegistered['geneListMasked']):
    if np.nansum(digitalGeneMatrixNaN[geneIdx,:]) > 0:
        plt.imshow(templateData['wholeBrainAnnotEdges'], cmap='gray_r')
        plt.scatter(templateDigitalSpots[:,0], templateDigitalSpots[:,1], c=digitalGeneMatrixNaN[geneIdx,:], marker='.', cmap='Reds')
        plt.title(actGene)
        plt.axis('off')
        plt.savefig(os.path.join(derivatives,f'geneExpression{actGene}.png'), bbox_inches='tight', dpi=300)
        plt.show()
        
#%% 
cellTypeJson = open(os.path.join('/','home','zjpeters','Documents','stanly','data','cellTypeMarkers.json'),)
cellTypeGeneLists = json.load(cellTypeJson)['cellTypes']

for i in range(len(cellTypeGeneLists)):
    try:
        neuCells = stanly.selectSpotsWithGeneList(processedSample,cellTypeGeneLists[i]['topGeneList'], threshold=0.9)
        plt.imshow(processedSample['tissueProcessed'], cmap='gray')
        plt.scatter(neuCells[1][:,0],neuCells[1][:,1])
        plt.title(f'{cellTypeGeneLists[i]["name"]}')
        plt.show()

    except TypeError:
        continue    
#%% display single gene
geneListSorted = np.sort(sampleRegistered['geneListMasked'])
# ast: Aqp4, end: Flt1, mic: Csf1r, neu: maybe Vipr1/Vipr2, opc: Pdgfra, Pcdh15
geneOfInterest='Pcdh15'
geneIdx = sampleRegistered['geneListMasked'].index(geneOfInterest)
plt.imshow(templateData['wholeBrainAnnotEdges'], cmap='gray_r')
plt.scatter(templateDigitalSpots[:,0], templateDigitalSpots[:,1], c=digitalGeneMatrixNaN[geneIdx,:], marker='.', cmap='Reds')
plt.title(geneOfInterest)
plt.show()

#%% display cell density
# 114,80 is 1/10 of template image size
nX,nY = 114,80
xBinEdges = np.linspace(0,templateData['wholeBrain'].shape[1],nX+1)
yBinEdges = np.linspace(0,templateData['wholeBrain'].shape[0],nY+1)
density, yEdges, xEdges = np.histogram2d(sampleRegistered['maskedTissuePositionList'][:,0],sampleRegistered['maskedTissuePositionList'][:,1], bins=(xBinEdges, yBinEdges))

plt.imshow(templateData['wholeBrain'])
plt.pcolormesh(yEdges, xEdges, density.T, cmap='rainbow', alpha=0.7)
plt.colorbar()
plt.imshow(templateData['wholeBrainAnnotEdges'], cmap='gray', alpha=1)
# plt.suptitle('Regional cell density',fontsize=14, horizontalalignment='center')
plt.title('Number of cells per square pixel')
plt.axis('off')
plt.show()

#%% display cell density per gene

for geneIdx,actGene in enumerate(sampleRegistered['geneListMasked']):
    if np.nansum(digitalGeneMatrixNaN[geneIdx,:]) > 0:
        actCellCoor = sampleRegistered['maskedTissuePositionList'][np.squeeze(np.array(sampleRegistered['geneMatrixMasked'][geneIdx,:].todense() > 0)),:]
        density, yEdges, xEdges = np.histogram2d(actCellCoor[:,0],actCellCoor[:,1], bins=(xBinEdges, yBinEdges))
        plt.imshow(templateData['wholeBrainAnnotEdges'])
        plt.pcolormesh(yEdges, xEdges, density.T, cmap='rainbow', alpha=0.7)
        plt.colorbar()
        plt.imshow(templateData['wholeBrainAnnotEdges'], cmap='gray', alpha=1)
        # plt.suptitle('Regional cell density',fontsize=14, horizontalalignment='center')
        plt.title(f'Density of cells expressing {actGene} per $100\mu m^2$')
        plt.axis('off')
        plt.savefig(os.path.join(derivatives,f'cellDensity{actGene}.png'), bbox_inches='tight', dpi=300)
        plt.show()



#%% try clustering on test sample
fullyConnectedEdges = []
sampleToCluster = processedSample
for i in range(digitalGeneMatrix.shape[1]):
    for j in range(digitalGeneMatrix.shape[1]):
        fullyConnectedEdges.append([i,j])
        
fullyConnectedEdges = np.array(fullyConnectedEdges,dtype='int32')
fullyConnectedEdges = np.unique(np.sort(fullyConnectedEdges, axis=1),axis=0)

#%% calculate cosine sim for single sample

start_time = time.time()
sampleToClusterGeneMatrix = np.array(digitalGeneMatrix,dtype='float32')
adjacencyDataControl = [stanly.cosineSimOfConnection(sampleToClusterGeneMatrix,I, J) for I,J in fullyConnectedEdges]
print("--- %s seconds ---" % (time.time() - start_time))  

with open(os.path.join(derivatives,f'adjacencyDataFor{sampleToCluster["sampleID"]}DigitalSpots.csv'), 'w', encoding='UTF8') as f:
    writer = csv.writer(f)
    writer.writerow(adjacencyDataControl) 
    
#%% create laplacian for digital gene matrix
WgeneMatrix= np.zeros([nDigitalSpots,nDigitalSpots],dtype='float32')
for idx, actCS in enumerate(adjacencyDataControl):
    WgeneMatrix[fullyConnectedEdges[idx,0],fullyConnectedEdges[idx,1]] = float(actCS)
    WgeneMatrix[fullyConnectedEdges[idx,1],fullyConnectedEdges[idx,0]] = float(actCS)
# W = sp_sparse.coo_matrix((np.array(adjacencyDataControl), (nnEdgeList[:,0],nnEdgeList[:,1])), shape=(nnControlSortedIdx.shape[0],nnControlSortedIdx.shape[0]), dtype='float32')
# W = W.todense()
WgeneMatrix = (WgeneMatrix - WgeneMatrix.min())/(WgeneMatrix.max() - WgeneMatrix.min())
WgeneMatrix[WgeneMatrix==1] = 0
DgeneMatrix = np.diag(sum(WgeneMatrix))
LgeneMatrix = DgeneMatrix - WgeneMatrix
eigvalgeneMatrix,eigvecgeneMatrix = np.linalg.eig(LgeneMatrix)
eigvalgeneMatrixSort = np.sort(np.real(eigvalgeneMatrix))[::-1]
eigvalgeneMatrixSortIdx = np.argsort(np.real(eigvalgeneMatrix))[::-1]
eigvecgeneMatrixSort = np.real(eigvecgeneMatrix[:,eigvalgeneMatrixSortIdx])

# run k means 
clusterK = 15
clusters = KMeans(n_clusters=clusterK, init='random', n_init=300, tol=1e-8,).fit(np.real(eigvecgeneMatrixSort[:,0:clusterK]))
plt.imshow(sampleRegistered['tissueRegistered'],cmap='gray')
plt.scatter(templateDigitalSpots[:,0], templateDigitalSpots[:,1],c=clusters.labels_,cmap='Set2')

#%% run sillhoutte analysis on control clustering

clusterRange = np.array(range(26,50))

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
    cluster_labels = clusters.fit_predict(np.real(eigvecgeneMatrixSort[:,0:actK]))

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(np.real(eigvecgeneMatrixSort[:,0:actK]), cluster_labels)
    print(
        "For n_clusters =",
        actK,
        "The average silhouette_score is :",
        silhouette_avg,
    )

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(np.real(eigvecgeneMatrixSort[:,0:actK]), cluster_labels)

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
    ax2.imshow(sampleRegistered['tissueRegistered'],cmap='gray_r')
    ax2.scatter(templateDigitalSpots[:,0], templateDigitalSpots[:,1],c=colors,cmap='tab20b')
    ax2.set_title("The visualization of the clustered merfish data.")
    ax2.axis('off')

    plt.suptitle(
        "Silhouette analysis for KMeans clustering on merfish sample data with n_clusters = %d"
        % actK,
        fontsize=14,
        fontweight="bold",
    )
    plt.savefig(os.path.join(derivatives,f'clusteringAndSilhouetteMerfish{actK}.png'), bbox_inches='tight', dpi=300)
    plt.show()
    
    

