#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 17:53:01 2022

@author: zjpeters
"""
#%% create moran's i calculation
import sklearn
# kSpots = 7

# equation for moran's I
#########################################################################################
# (N/W) (sigm(N,i=1)sigma(N,j=1) wij(xi - xmean)(xj-xmean))/(sigma(N,i=1) (xi-xmean)^2) #
# N = nDigitalSpots OR kSpots ?
# xi = gene count at spot xi
# xj = gene count at spot xj from nearest neighbors of xi
# xmean = mean gene count across all spots (including 0?)
# 
#########################################################################################

kSpots = 7

# consider whether digital moran's needs to be compared to individual visium moran's I

# first need to calculate an inverse distance matrix for the spots being used
# divide by 0 error is fine for now, replaces with inf, can deal with error message
spotInverseCdistSM = 1/cdist(inTissueTemplateSpots, inTissueTemplateSpots, 'euclidean')
# can now index necessary info from the inverse distance matrix with the nearest neighbor list to create weight matrix
spotCdistSM = cdist(inTissueTemplateSpots, inTissueTemplateSpots, 'euclidean')
sortedSpotCdistSM = np.sort(spotCdistSM, axis=0)
sortedSpotCdistSMidx = np.argsort(spotCdistSM, axis=0)
digitalSpotCdist = sortedSpotCdistSM[1:kSpots+1]
digitalSpotNNidx = sortedSpotCdistSMidx[1:kSpots+1]
# spotNNIdx gives the index of the top kSpots nearest neighbors for each digital spot
spotMeanCdist = np.mean(np.transpose(digitalSpotCdist))


spotNNIdx = []
for NNs in enumerate(np.transpose(digitalSpotCdist)):
    spotMeanCdist = np.mean(NNs[1])
    # changing from 20 to 27 for digital calculation, since that's ~2 spot centers away
    if spotMeanCdist < 27:
        spotNNIdx.append(digitalSpotNNidx[:,NNs[0]])

    else:
        # should probably change this from 0s to something like -1
        spotNNIdx.append(np.transpose(np.zeros([kSpots],dtype=int)))

spotNNIdx = np.array(spotNNIdx)
    
for nOfGenesChecked,actGene in enumerate(geneListFromTxt):
    # geneToSearch = actGene
    
    # allSamplesDigitalNearestNeighbors = []
    # digitalSamples = []
    digitalSamplesControl = []
    digitalSamplesExperimental = []
    # meanDigitalSample = np.zeros([nDigitalSpots,1])
    meanDigitalControls = np.zeros([nDigitalSpots])
    meanDigitalExperimentals = np.zeros([nDigitalSpots])
    nTestedSamples = 0
    nControls = 0
    nExperimentals = 0
    for actSample in range(nTotalSamples):
        try:
            geneIndex = allSamplesToAllen[actSample]['geneListMasked'].index(actGene)
            spotCheck = np.count_nonzero(allSamplesToAllen[actSample]['filteredFeatureMatrixMasked'][geneIndex,:])
    
            if spotCheck < 15:
                continue
            actNN,actCDist = findDigitalNearestNeighbors(inTissueTemplateSpots, allSamplesToAllen[actSample]['maskedTissuePositionList'], kSpots)
            geneCount = np.zeros([nDigitalSpots,kSpots])
            for spots in enumerate(actNN):
                if ~np.all(spots[1]):
                    geneCount[spots[0]] = 0
                else:
                    geneCount[spots[0]] = allSamplesToAllen[actSample]['filteredFeatureMatrixMasked'][geneIndex,actNN[spots[0]]]
                    
            spotCount = np.nanmean(geneCount, axis=1)
            meanSpotCount = np.nanmean(geneCount)
            nTestedSamples += 1
            if truncExperiment['experimental-group'][actSample] == 0:
                # print("Slice is control")
                digitalSamplesControl.append(spotCount)
                meanDigitalControls += spotCount
                # this gives the number of control samples with more than 15 spots containing the gene
                nControls += 1
            elif truncExperiment['experimental-group'][actSample] == 1:
                # print("Slice is experimental")
                digitalSamplesExperimental.append(spotCount)
                meanDigitalExperimentals += spotCount
                # this gives the number of experimental samples with more than 15 spots containing the gene
                nExperimentals += 1
                
            else:
                continue
            
        except:
            continue
    if spotCheck < 15:
        continue
    digitalSamplesControl = np.array(digitalSamplesControl, dtype=float).squeeze()
    digitalSamplesExperimental = np.array(digitalSamplesExperimental, dtype=float).squeeze()
    meanDigitalControls = meanDigitalControls / nControls
    meanDigitalExperimentals = meanDigitalExperimentals / nExperimentals
    
    xMeanControls = np.mean(meanDigitalControls)
    xMeanExperimentals = np.mean(meanDigitalExperimentals)
    xMean = np.mean([meanDigitalControls,meanDigitalExperimentals])
    wij = np.zeros([nDigitalSpots,kSpots])
    # xiDeltaSquaredControls = np.zeros([nDigitalSpots,kSpots])
    # xiDeltaSquaredExperimentals = np.zeros([nDigitalSpots,kSpots])
    xjDeltaControls = np.zeros([nDigitalSpots,kSpots])
    xjDeltaExperimentals = np.zeros([nDigitalSpots,kSpots])
    xiDeltaControls = np.zeros([nDigitalSpots])
    xiDeltaExperimentals = np.zeros([nDigitalSpots])
    xiDeltaSquaredControls = np.zeros([nDigitalSpots])
    xiDeltaSquaredExperimentals = np.zeros([nDigitalSpots])
    xjDeltaSquaredControls = np.zeros([nDigitalSpots,kSpots])
    xjDeltaSquaredExperimentals = np.zeros([nDigitalSpots,kSpots])
    for NNs in enumerate(spotNNIdx):
        
        xiDeltaControls[NNs[0]] = meanDigitalControls[NNs[0]] - xMeanControls
        xiDeltaExperimentals[NNs[0]] = meanDigitalExperimentals[NNs[0]] - xMeanExperimentals
        xiDeltaSquaredControls[NNs[0]] = np.square(xiDeltaControls[NNs[0]])
        xiDeltaSquaredExperimentals[NNs[0]] = np.square(xiDeltaExperimentals[NNs[0]])
    # retrying below code
        # for j in enumerate(NNs[1]):
        #     xjDeltaControls = meanDigitalControls[j[1]] - xMeanControls
        #     xjDeltaExperimentals = meanDigitalExperimentals[j[1]] - xMeanExperimentals
        #     spatialLagControls[NNs[0],j[0]] = ( xiDeltaControls * xjDeltaControls * spotInverseCdistSM[NNs[0],j[1]] ) / xiDeltaSquaredControls
        #     spatialLagExperimentals[NNs[0],j[0]] = ( xiDeltaExperimentals * xjDeltaExperimentals * spotInverseCdistSM[NNs[0],j[1]] ) / xiDeltaSquaredExperimentals
        #     # spatialLagExperimentals[NNs[0],j[0]] = xiDeltaExperimentals * xjDeltaExperimentals * spotInverseCdistSM[NNs[0],j[1]]
        #     wij[NNs[0],j[0]] = spotInverseCdistSM[NNs[0],j[1]]
            
        for j in enumerate(NNs[1]):
            # xjDeltaControls = meanDigitalControls[j[1]] - xMeanControls
            # xjDeltaExperimentals = meanDigitalExperimentals[j[1]] - xMeanExperimentals
            xjDeltaControls[NNs[0],j[0]] = meanDigitalControls[j[1]] - xMeanControls
            xjDeltaExperimentals[NNs[0],j[0]] = meanDigitalExperimentals[j[1]] - xMeanExperimentals
            xjDeltaSquaredControls[NNs[0]] = np.square(xjDeltaControls[NNs[0]])
            xjDeltaSquaredExperimentals[NNs[0]] = np.square(xjDeltaExperimentals[NNs[0]])
            # spatialLagExperimentals[NNs[0],j[0]] = xiDeltaExperimentals * xjDeltaExperimentals * spotInverseCdistSM[NNs[0],j[1]]
            wij[NNs[0],j[0]] = spotCdistSM[NNs[0],j[1]]
    # row standardize wij        
    normalizedWij = sklearn.preprocessing.normalize(wij, norm="l1")
    # local moran's i
    m2Controls = np.sum(xjDeltaSquaredControls) / (nDigitalSpots - 1)
    m2Experimentals = np.sum(xjDeltaSquaredExperimentals) / (nDigitalSpots - 1)
    IiControls = (xiDeltaControls / m2Controls) * np.sum(np.multiply(normalizedWij, xjDeltaControls))
    IiExperimentals = (xiDeltaExperimentals / m2Experimentals) * np.sum(np.multiply(normalizedWij, xjDeltaExperimentals))
    
    # below still isn't quite right?
    globalIControls = np.sum(IiControls / nDigitalSpots)
    globalIExperimentals = np.sum(IiExperimentals / nDigitalSpots)
    localIMin = np.min([IiControls,IiExperimentals])
    localIMax = np.max([IiControls,IiExperimentals])
    
                    
    maxGeneCount = np.max([meanDigitalControls,meanDigitalExperimentals])
    
    plt.imshow(bestSampleToTemplate['visiumTransformed'])
    plt.scatter(inTissueTemplateSpots[:,0],inTissueTemplateSpots[:,1], c=np.array(meanDigitalControls), alpha=0.8, vmin=0,vmax=maxGeneCount,plotnonfinite=False)
    plt.title(f'Mean gene count for {actGene}, control')
    plt.colorbar()
    plt.show()
    
    plt.imshow(bestSampleToTemplate['visiumTransformed'])
    plt.scatter(inTissueTemplateSpots[:,0],inTissueTemplateSpots[:,1], c=np.array(meanDigitalExperimentals), alpha=0.8, vmin=0,vmax=maxGeneCount,plotnonfinite=False)
    plt.title(f'Mean gene count for {actGene}, sleep deprivation')
    plt.colorbar()
    plt.show()

    if localIMin > 0:
        plt.imshow(bestSampleToTemplate['visiumTransformed'])
        plt.scatter(inTissueTemplateSpots[:,0],inTissueTemplateSpots[:,1], c=np.array(IiControls), cmap='Reds',alpha=0.8,vmin=0, vmax=localIMax)
        plt.title(f'testing Morans I for {actGene}, non-sleep dep, global I: {globalIControls}')
        plt.colorbar()
        plt.show()
        
        plt.imshow(bestSampleToTemplate['visiumTransformed'])
        plt.scatter(inTissueTemplateSpots[:,0],inTissueTemplateSpots[:,1], c=np.array(IiExperimentals), cmap='Reds',alpha=0.8,vmin=0, vmax=localIMax)
        plt.title(f'testing Morans I for {actGene}, sleep dep, global I: {globalIExperimentals}')
        plt.colorbar()
        plt.show()

    elif localIMax < 0:
        plt.imshow(bestSampleToTemplate['visiumTransformed'])       
        plt.scatter(inTissueTemplateSpots[:,0],inTissueTemplateSpots[:,1], c=np.array(IiControls), cmap='Blues_r',alpha=0.8,vmin=localIMin, vmax=0)
        plt.title(f'testing Morans I for {actGene}, non-sleep dep, global I: {globalIControls}')
        plt.colorbar()
        plt.show()
        
        plt.imshow(bestSampleToTemplate['visiumTransformed'])
        plt.scatter(inTissueTemplateSpots[:,0],inTissueTemplateSpots[:,1], c=np.array(IiExperimentals), cmap='Blues_r',alpha=0.8,vmin=localIMin, vmax=0)
        plt.title(f'testing Morans I for {actGene}, sleep dep, global I: {globalIExperimentals}')
        plt.colorbar()
        plt.show()
        
    else:
        zeroCenteredCmap = mcolors.TwoSlopeNorm(0, localIMin, vmax=localIMax)
        plt.imshow(bestSampleToTemplate['visiumTransformed'])
        plt.scatter(inTissueTemplateSpots[:,0],inTissueTemplateSpots[:,1], c=np.array(IiControls), cmap='seismic',alpha=0.8,norm=zeroCenteredCmap,plotnonfinite=False)
        plt.title(f'testing Morans I for {actGene}, non-sleep dep, global I: {globalIControls}')
        plt.colorbar()
        plt.show()
        
        plt.imshow(bestSampleToTemplate['visiumTransformed'])
        plt.scatter(inTissueTemplateSpots[:,0],inTissueTemplateSpots[:,1], c=np.array(IiExperimentals), cmap='seismic',alpha=0.8,norm=zeroCenteredCmap,plotnonfinite=False)
        plt.title(f'testing Morans I for {actGene}, sleep dep, global I: {globalIExperimentals}')
        plt.colorbar()
        plt.show()
        
    spotLocalIDelta = IiControls - IiExperimentals
    spotLocalIDeltaMin = np.min(spotLocalIDelta)
    spotLocalIDeltaMax = np.max(spotLocalIDelta)

    if spotLocalIDeltaMin > 0:
        plt.imshow(bestSampleToTemplate['visiumTransformed'])
        plt.scatter(inTissueTemplateSpots[:,0],inTissueTemplateSpots[:,1], c=np.array(spotLocalIDelta), cmap='Reds',alpha=0.8,vmin=0, vmax=localIMax)
        plt.title(f'Local Morans I difference for {actGene}')
        plt.colorbar()
        plt.show()
        

    elif spotLocalIDeltaMax < 0:
        plt.imshow(bestSampleToTemplate['visiumTransformed'])       
        plt.scatter(inTissueTemplateSpots[:,0],inTissueTemplateSpots[:,1], c=np.array(spotLocalIDelta), cmap='Blues_r',alpha=0.8,vmin=localIMin, vmax=0)
        plt.title(f'Local Morans I difference for {actGene}')
        plt.colorbar()
        plt.show()

        
    else:
        try:
            zeroCenteredCmap = mcolors.TwoSlopeNorm(0, vmin=spotLocalIDeltaMin, vmax=spotLocalIDeltaMax)
            plt.imshow(bestSampleToTemplate['visiumTransformed'])
            plt.scatter(inTissueTemplateSpots[:,0],inTissueTemplateSpots[:,1], c=np.array(spotLocalIDelta), cmap='seismic',alpha=0.8,norm=zeroCenteredCmap,plotnonfinite=False)
            plt.title(f'Local Morans I difference for {actGene}')
            plt.colorbar()
            plt.show()

        except:
            continue
# sumOfWeightedDifferencesControls = np.sum(np.multiply(normalizedWij, spatialLagControls),axis=1)
# sumOfWeightedDifferencesExperimentals = np.sum(np.multiply(normalizedWij, spatialLagExperimentals),axis=1)

# Icontrols = np.divide(sumOfWeightedDifferencesControls, xiDeltaSquaredControls)

# Iexperimentals = np.divide(sumOfWeightedDifferencesExperimentals, xiDeltaSquaredExperimentals)

# sumOfWeightedDifferencesControls = np.sum(np.multiply(normalizedWij, spatialLagControls))
# sumOfWeightedDifferencesExperimentals = np.sum(np.multiply(normalizedWij, spatialLagExperimentals))

# nW = kSpots / np.sum(normalizedWij)
# Icontrols = nW * np.divide(sumOfWeightedDifferencesControls, sum(xiDeltaSquaredControls))

# Iexperimentals = nW * np.divide(sumOfWeightedDifferencesExperimentals, sum(xiDeltaSquaredExperimentals))

# plt.imshow(bestSampleToTemplate['visiumTransformed'])
# plt.scatter(inTissueTemplateSpots[:,0],inTissueTemplateSpots[:,1], c=np.array(Icontrols), cmap='seismic',alpha=0.8)
# plt.title(f'testing Morans I for {actGene}')
# plt.colorbar()
# plt.show()

