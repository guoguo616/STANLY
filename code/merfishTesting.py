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
import scipy
# import pandas as pd
# from skimage.transform import rescale, rotate, resize
# import itk
import sys
sys.path.insert(0, "/home/zjpeters/Documents/stanly/code")
import stanly
from glob import glob
# from skimage import io, filters, color, feature, morphology
import csv
# import cv2
# from skimage.exposure import match_histograms
# import scipy.sparse as sp_sparse
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

# # import and align sample data to allen ccf

# sampleData = stanly.importMerfishData(sourcedata, derivatives)
# processedSample = stanly.processMerfishData(sampleData, templateData, 210, derivatives)

# #%% split two samples from one image using lasso tool

# selectorRight = stanly.SelectUsingLasso(processedSample, 'rightHem')
# #%%
# selectorRight.outputMaskedSpots()
# selectorRight.outputMaskedImage(processedSample)
# rightHem = selectorRight.outputMaskedSample(processedSample)
# # = selector.maskedSpots

# # totalSpotCount = 0


# #%%
# selectorLeft = stanly.SelectUsingLasso(processedSample,'leftHem')

# #%%
# selectorLeft.outputMaskedSpots()
# selectorLeft.outputMaskedImage(processedSample)
# selectorLeft.flip()
# leftHem = selectorLeft.outputMaskedSample(processedSample)

rightHem = stanly.loadProcessedMerfishSample('/media/zjpeters/Samsung_T5/merscope/derivatives/Slide1_Apr24_rightHem')
leftHem = stanly.loadProcessedMerfishSample('/media/zjpeters/Samsung_T5/merscope/derivatives/Slide1_Apr24_leftHem')
#%% register processed samples
bestSampleToTemplate = stanly.runANTsToAllenRegistration(rightHem, templateData, hemisphere='rightHem')

#%%
processedSamples = {}
processedSamples[0] = rightHem
processedSamples[1] = leftHem

experimentalResults = {}
for actSample in range(len(processedSamples)):
    sampleRegistered = stanly.runANTsInterSampleRegistration(processedSamples[actSample], processedSamples[0])
    experimentalResults[actSample] = sampleRegistered

allSamplesToAllen = {}
for actSample in range(len(experimentalResults)):
    regSampleToTemplate = stanly.applyAntsTransformations(experimentalResults[actSample], bestSampleToTemplate, templateData, hemisphere='rightHem')
    allSamplesToAllen[actSample] = regSampleToTemplate

#%% create digital spots and find nearest neighbors
wholeBrainSpotSize = 10
kSpots = 7
templateDigitalSpots = stanly.createDigitalSpots(allSamplesToAllen[0], wholeBrainSpotSize)

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
nSampleExperimental = 1
nSampleControl = 1
nGenesInList = len(allSampleGeneList)

for sampleIdx, actSample in enumerate(allSamplesToAllen):
    allSamplesToAllen[sampleIdx]['allSampleGeneList'] = allSampleGeneList 
    sortedIdxList = np.zeros(nGenesInList,dtype='int32')
    for sortedIdx, actGene in enumerate(allSampleGeneList):
        sortedIdxList[sortedIdx] = allSamplesToAllen[sampleIdx]['geneListMasked'].index(actGene)
    allSamplesToAllen[sampleIdx]['geneMatrixMaskedSorted'] = allSamplesToAllen[sampleIdx]['geneMatrixMasked'][sortedIdxList,:].astype('int32')
    allSamplesToAllen[sampleIdx].pop('geneMatrixMasked')
    allSamplesToAllen[sampleIdx].pop('geneListMasked')

#%% first test using Sidak correction

start_time = time.time()
desiredPval = 0.05
alphaSidak = 1 - np.power((1 - desiredPval),(1/(len(allSampleGeneList))))
rankList = np.arange(1,nDigitalSpots+1)
bhCorrPval = (rankList/(nDigitalSpots*len(allSampleGeneList)))*desiredPval
bonCorrPval = desiredPval/(len(allSampleGeneList))
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
        geneIndex = nOfGenesChecked
        geneCount = np.zeros([nDigitalSpots,kSpots])
        for spots in enumerate(allSamplesToAllen[actSample]['digitalSpotNearestNeighbors']):
            if np.any(spots[1] < 0):
                geneCount[spots[0]] = np.nan
            else:
                spotij = np.zeros([7,2], dtype='int32')
                spotij[:,1] = np.asarray(spots[1], dtype='int32')
                spotij[:,0] = geneIndex
                
                geneCount[spots[0]] = allSamplesToAllen[actSample]['geneMatrixMaskedSorted'][spotij[:,0],spotij[:,1]]
                
        spotCount = np.nanmean(geneCount, axis=1)
        nTestedSamples += 1
        if actSample == 0:
            digitalSamplesControl[:,startControl:stopControl] = geneCount
            startControl += kSpots
            stopControl += kSpots
            nControls += 1
        elif actSample == 1:
            digitalSamplesExperimental[:,startExperimental:stopExperimental] = geneCount
            startExperimental += kSpots
            stopExperimental += kSpots
            nExperimentals += 1
        else:
            continue
    
    digitalSamplesControl = np.array(digitalSamplesControl, dtype='float32').squeeze()
    digitalSamplesExperimental = np.array(digitalSamplesExperimental, dtype='float32').squeeze()
    
    # this will check that at least a certain number of spots show expression for the gene of interest
    checkControlSamples = np.count_nonzero(digitalSamplesControl,axis=1)
    checkExperimentalSamples = np.count_nonzero(digitalSamplesExperimental,axis=1)
    checkAllSamples = checkControlSamples & checkExperimentalSamples > 20
    if sum(checkAllSamples) < 1:
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
        mulCompResults = actPvals < alphaSidak
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
            meanDigitalControl = np.nanmean(digitalSamplesControl,axis=1)
            meanDigitalExperimental = np.nanmean(digitalSamplesExperimental,axis=1)
            finiteMin = np.nanmin(actTtest[0])
            finiteMax = np.nanmax(actTtest[0])
            maxGeneCount = np.nanmax([medianDigitalControl,medianDigitalExperimental])
            #Plot data
            hcMax = np.nanmax(meanDigitalControl)
            sorMax = np.nanmax(meanDigitalExperimental)
            plotMax = np.max([hcMax, sorMax])
            fig, axs = plt.subplots(1,3)
            # display mean gene count for control group            
            plt.axis('off')
            axs[0].imshow(bestSampleToTemplate['tissueRegistered'],cmap='gray',aspect="equal")
            axs[0].scatter(templateDigitalSpots[:,0],templateDigitalSpots[:,1], c=np.array(meanDigitalControl), alpha=0.5,plotnonfinite=False,cmap='Reds',marker='.', vmin=0, vmax=plotMax)
            # axs[0].imshow(template['leftHemAnnotEdges'], cmap='gray_r')
            axs[0].set_title('HC')
            axs[0].axis('off')
            # display mean gene count for experimental group
            axs[1].imshow(bestSampleToTemplate['tissueRegistered'],cmap='gray',aspect="equal")
            axs[1].scatter(templateDigitalSpots[:,0],templateDigitalSpots[:,1], c=np.array(meanDigitalExperimental), alpha=0.5,plotnonfinite=False,cmap='Reds',marker='.', vmin=0, vmax=plotMax)
            axs[1].set_title('SOR')
            axs[1].axis('off')
            # display t-statistic for exp > control
            axs[2].scatter(templateDigitalSpots[:,0],templateDigitalSpots[:,1], c=np.array(actTstats), cmap='seismic',alpha=0.5,vmin=-4,vmax=4,plotnonfinite=False,marker='.')
            axs[2].imshow(bestSampleToTemplate['tissueRegistered'],cmap='gray',aspect="equal")
            axs[2].set_title(actGene, style='italic')
            axs[2].axis('off')
            plt.savefig(os.path.join(derivatives,f'tStatGeneCount{actGene}.png'), bbox_inches='tight', dpi=300)
            plt.show()

timestr = time.strftime("%Y%m%d-%H%M%S")
with open(os.path.join(derivatives,f'listOfSigGenesSidakPvalues{timestr}.csv'), 'w', encoding='UTF8') as f:
    writer = csv.writer(f)
    for i in sigGenesWithPvals:
        writer.writerow(i)
        
with open(os.path.join(derivatives,f'listOfSigGenesSidakTstatistics{timestr}.csv'), 'w', encoding='UTF8') as f:
    writer = csv.writer(f)
    for i in sigGenesWithTstats:
        writer.writerow(i)
        
print("--- %s seconds ---" % (time.time() - start_time))


"""

"""
###############################################################################
#%% test digital spot creation using merfish to perform clustering
wholeBrainSpotSize = 10
templateDigitalSpots = stanly.createDigitalSpots(bestSampleToTemplate, wholeBrainSpotSize)
nDigitalSpots = len(templateDigitalSpots)
nGenesInList = len(bestSampleToTemplate['geneListMasked'])
kSpots = 7
#actNN, actCDist = stanly.findDigitalNearestNeighbors(templateDigitalSpots, bestSampleToTemplate['maskedTissuePositionList'], kSpots, wholeBrainSpotSize)
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
nSampleExperimental = 1 # sum(experiment['experimental-group'])
nSampleControl = 1 # len(experiment['experimental-group']) - nSampleExperimental
nGenesInList = len(allSampleGeneList)actSpotCdist

for sampleIdx, actSample in enumerate(allSamplesToAllen):
    allSamplesToAllen[sampleIdx]['allSampleGeneList'] = allSampleGeneList 
    sortedIdxList = np.zeros(nGenesInList,dtype='int32')
    for sortedIdx, actGene in enumerate(allSampleGeneList):
        sortedIdxList[sortedIdx] = allSamplesToAllen[sampleIdx]['geneListMasked'].index(actGene)
    allSamplesToAllen[sampleIdx]['geneMatrixMaskedSorted'] = allSamplesToAllen[sampleIdx]['geneMatrixMasked'][sortedIdxList,:].astype('int32')
    allSamplesToAllen[sampleIdx].pop('geneMatrixMasked')
    allSamplesToAllen[sampleIdx].pop('geneListMasked')
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
    
    

