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

#%% output image of arc spots for each subject before registration
actGene = 'Camk2n1'
for i, regSample in enumerate(processedSamples):
    try:
        geneIndex = processedSamples[i]['geneListMasked'].index(actGene)
    except(ValueError):
        print(f'{actGene} not in dataset')
        continue
    actSpots = processedSamples[i]['filteredFeatureMatrixLog2'][geneIndex, :]
    plt.imshow( processedSamples[i]['tissueProcessed'], cmap='gray')
    plt.scatter(processedSamples[i]['processedTissuePositionList'][:,0],processedSamples[i]['processedTissuePositionList'][:,1], c=np.array(actSpots.todense()), alpha=0.8, cmap='Reds', marker='.')
    plt.title(f'Gene count for {actGene} in {processedSamples[i]["sampleID"]}')
    plt.colorbar()
    plt.savefig(os.path.join(derivatives,f'geneCount{actGene}{processedSamples[i]["sampleID"]}Registered.png'), bbox_inches='tight', dpi=300)
    plt.show()
#%% can now use this gene list to loop over expressed genes 
# 'Arc','Egr1','Lars2','Ccl4'
# testGeneList = ['Arc','Egr1','Rpl21']
# caudoputamenGeneList = ['Adora2a','Drd2','Pde10a','Drd1','Scn4b','Gpr6','Ido1','Adcy5','Rasd2','Meis2','Lars2','Ccl4']
# allocortexGeneList = ['Nptxr','Lmo3','Slc30a3','Syn2','Snca','Ccn3','Bmp3','Olfm1','Ldha','Tafa2']
# fibertractsGeneList = ['Plp1','Mag','Opalin','Cnp','Trf','Cldn11','Cryab','Mobp','Qdpr','Sept4']
# hippocampalregionGeneList = ['Wipf3','Cabp7','Cnih2','Gria1','Ptk2b','Cebpb','Nr3c2','Lct','Arhgef25','Epha7']
# hypothalamusGeneList = ['Gpx3','Resp18','AW551984','Minar2','Nap1l5','Gabrq','Pcbd1','Sparc','Vat1','6330403K07Rik']
# neocortexGeneList = ['1110008P14Rik','Ccl27a','Mef2c','Tbr1','Cox8a','Snap25','Nrgn','Vxn','Efhd2','Satb2']
# striatumlikeGeneList = ['Hap1','Scn5a','Pnck','Ahi1','Snhg11','Galnt16','Pnmal2','Baiap3','Ly6h','Meg3']
# thalamusGeneList = ['Plekhg1','Tcf7l2','Ntng1','Ramp3','Rora','Patj','Rgs16','Nsmf','Ptpn4','Rab37']
# testGeneList = testGeneList + caudoputamenGeneList + allocortexGeneList + fibertractsGeneList + hippocampalregionGeneList + hypothalamusGeneList + neocortexGeneList + striatumlikeGeneList + thalamusGeneList

wholeBrainSpotSize = 15
kSpots = 7
templateDigitalSpots = stanly.createDigitalSpots(allSamplesToAllen[4], wholeBrainSpotSize)

allSampleGeneList = allSamplesToAllen[0]['geneListMasked']
for i, regSample in enumerate(allSamplesToAllen):        
    actNN, actCDist = stanly.findDigitalNearestNeighbors(templateDigitalSpots, allSamplesToAllen[i]['maskedTissuePositionList'], kSpots, wholeBrainSpotSize)
    allSamplesToAllen[i]['digitalSpotNearestNeighbors'] = np.asarray(actNN, dtype=int)
    # creates a list of genes present in all samples
    if i == 0:
        continue
    else:
        allSampleGeneList = set(allSampleGeneList) & set(allSamplesToAllen[i]['geneListMasked'])

start_time = time.time()

nDigitalSpots = len(templateDigitalSpots)
nSampleExperimental = sum(experiment['experimental-group'])
nSampleControl = len(experiment['experimental-group']) - nSampleExperimental

alphaSidak = 1 - np.power((1 - 0.05),(1/nDigitalSpots))
# alphaSidak = 5e-8
# list(allSampleGeneList)[0:1000]

# geneList = stanly.loadGeneListFromCsv('/home/zjpeters/Documents/visiumalignment/derivatives/dataFrom221221/listOfSigSleepDepGenes20221221-002356.csv')
sigGenes = []
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
        try:
            geneIndex = allSamplesToAllen[actSample]['geneListMasked'].index(actGene)
        except(ValueError):
            print(f'{actGene} not in dataset')
            continue

        geneCount = np.zeros([nDigitalSpots,kSpots])
        for spots in enumerate(allSamplesToAllen[actSample]['digitalSpotNearestNeighbors']):
            if np.all(spots[1] < 0):
                geneCount[spots[0]] = np.nan
            else:
                spotij = np.zeros([7,2], dtype=int)
                spotij[:,1] = np.asarray(spots[1], dtype=int)
                spotij[:,0] = geneIndex
                
                geneCount[spots[0]] = allSamplesToAllen[actSample]['filteredFeatureMatrixMasked'][spotij[:,0],spotij[:,1]]
                
        spotCount = np.nanmean(geneCount, axis=1)
        nTestedSamples += 1
        if experiment['experimental-group'][actSample] == 0:
            digitalSamplesControl[:,startControl:stopControl] = geneCount
            startControl += kSpots
            stopControl += kSpots
            nControls += 1
        elif experiment['experimental-group'][actSample] == 1:
            digitalSamplesExperimental[:,startExperimental:stopExperimental] = geneCount
            startExperimental += kSpots
            stopExperimental += kSpots
            nExperimentals += 1
            
        else:
            continue
    
    
    digitalSamplesControl = np.array(digitalSamplesControl, dtype=float).squeeze()
    digitalSamplesExperimental = np.array(digitalSamplesExperimental, dtype=float).squeeze()
    ############################
    # this will check that at least a certain number of spots show expression for the gene of interest #
    ##################
    checkControlSamples = np.count_nonzero(digitalSamplesControl,axis=1)
    checkExperimentalSamples = np.count_nonzero(digitalSamplesExperimental,axis=1)
    checkAllSamples = checkControlSamples & checkExperimentalSamples > 20
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
        allPvals = []
        actTtest = scipy.stats.ttest_ind(digitalSamplesExperimental,digitalSamplesControl, axis=1, nan_policy='propagate')
        actTstats = actTtest[0]
        actPvals = actTtest[1]
        mulCompResults = actPvals < alphaSidak
        # mulCompResults = multipletests(actTtest[1], 0.05, method='bonferroni', is_sorted=False)
        # fdrAlpha = mulCompResults[3].
        spotThr = 20 #0.05 * nDigitalSpots
        if sum(mulCompResults) > spotThr:
            sigGenes.append([actGene,sum(mulCompResults)])
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
            # if finiteMin > 0:
            #     finiteMin = -1
            # elif finiteMax < 0:
            #     finiteMax = 1
            # zeroCenteredCmap = mcolors.TwoSlopeNorm(vmin=-3, 0, vmax=3)
            # tTestColormap = zeroCenteredCmap(actTtest[0])
            maxGeneCount = np.nanmax([medianDigitalControl,medianDigitalExperimental])
            # display mean gene count for control group            
            fig = plt.figure()
            fig.add_subplot(1,3,1)
            plt.axis('off')
            plt.imshow(allSamplesToAllen[4]['visiumTransformed'],cmap='gray')
            plt.scatter(templateDigitalSpots[:,0],templateDigitalSpots[:,1], c=np.array(meanDigitalControl), alpha=0.8, vmin=0,vmax=maxGeneCount,plotnonfinite=False,cmap='Reds',marker='.')
            plt.title('NSD')

            # plt.savefig(os.path.join(derivatives,f'meanGeneCount{actGene}Control.png'), bbox_inches='tight', dpi=300)
            # plt.show()
            # display mean gene count for experimental group
            fig.add_subplot(1,3,2)
            plt.axis('off')
            plt.imshow(allSamplesToAllen[4]['visiumTransformed'],cmap='gray')
            expScatter = plt.scatter(templateDigitalSpots[:,0],templateDigitalSpots[:,1], c=np.array(meanDigitalExperimental), alpha=0.8, vmin=0,vmax=maxGeneCount,plotnonfinite=False,cmap='Reds',marker='.')
            plt.title('SD')
            fig.colorbar(expScatter,fraction=0.046, pad=0.04)

            fig.add_subplot(1,3,3)
            plt.axis('off')
            plt.imshow(allSamplesToAllen[4]['visiumTransformed'],cmap='gray')
            tStatScatter = plt.scatter(templateDigitalSpots[:,0],templateDigitalSpots[:,1], c=np.array(actTtest[0]), cmap='seismic',alpha=0.8,vmin=-4,vmax=4,plotnonfinite=False,marker='.')
            plt.title(f't-statistic for {actGene}')
            fig.colorbar(tStatScatter,fraction=0.046, pad=0.04)
            plt.savefig(os.path.join(derivatives,f'tStatGeneCount{actGene}SleepDep.png'), bbox_inches='tight', dpi=300)
            plt.show()
            
            # plt.colorbar()
            # plt.savefig(os.path.join(derivatives,f'meanGeneCount{actGene}SleepDep.png'), bbox_inches='tight', dpi=300)
            # plt.show()
            # display t statistics
            # fig.add_subplot(1,3,3)
            # plt.axis('off')
            # plt.imshow(bestSampleToTemplate['visiumTransformed'],cmap='gray')
            # plt.scatter(templateDigitalSpots[:,0],templateDigitalSpots[:,1], c=np.array(actTtest[0]), cmap='seismic',alpha=0.8,norm=zeroCenteredCmap,plotnonfinite=False,marker='.')
            # plt.title(f't-statistic for {actGene}' )   
            # plt.savefig(os.path.join(derivatives,f'tStatGeneCount{actGene}SleepDep.png'), bbox_inches='tight', dpi=300)
            # plt.show()

timestr = time.strftime("%Y%m%d-%H%M%S")
with open(os.path.join(derivatives,f'listOfSigSleepDepGenes{timestr}.csv'), 'w', encoding='UTF8') as f:
    writer = csv.writer(f)
    for i in sigGenes:
        writer.writerow(i)
print("--- %s seconds ---" % (time.time() - start_time))

#%% rewriting t stat section to output everything asked for, no matter significance, good for searching a gene list
wholeBrainSpotSize = 15
templateDigitalSpots = stanly.createDigitalSpots(allSamplesToAllen[4], wholeBrainSpotSize)
for i, regSample in enumerate(allSamplesToAllen):        
    actNN, actCDist = stanly.findDigitalNearestNeighbors(templateDigitalSpots, allSamplesToAllen[i]['maskedTissuePositionList'], kSpots, wholeBrainSpotSize)
    allSamplesToAllen[i]['digitalSpotNearestNeighbors'] = np.asarray(actNN, dtype=int)
    # creates a list of genes present in all samples
    if i == 0:
        continue
    else:
        allSampleGeneList = set(allSampleGeneList) & set(allSamplesToAllen[i]['geneListMasked'])

genesFromYann = ['Arc', 'Cirbp', 'Rbm3', 'Homer1', 'Pmch', 'Hcrt', 'Ttr', 'Bhlhe41','Marcksl1','Ep300','Nr4a1','Per1','Ube3a','Sh3gl1','Slc9a3r1', 'Marcksl1', 'Malat1', 'Hart']
start_time = time.time()

nDigitalSpots = len(templateDigitalSpots)

nSampleExperimental = sum(experiment['experimental-group'])
nSampleControl = len(experiment['experimental-group']) - nSampleExperimental

alphaSidak = 1 - np.power((1 - 0.05),(1/nDigitalSpots))
# alphaSidak = 5e-8
# list(allSampleGeneList)[0:1000]
geneList = allSampleGeneList
sigGenes = []
for nOfGenesChecked,actGene in enumerate(genesFromYann):
    digitalSamplesControl = np.zeros([nDigitalSpots,(nSampleControl * kSpots)])
    digitalSamplesExperimental = np.zeros([nDigitalSpots,(nSampleExperimental * kSpots)])
    startControl = 0
    stopControl = kSpots
    startExperimental = 0
    stopExperimental = kSpots
    nTestedSamples = 0
    nControls = 0
    nExperimentals = 0
    for actSample in range(nTotalSamples):
        try:
            geneIndex = allSamplesToAllen[actSample]['geneListMasked'].index(actGene)
        except(ValueError):
            print(f'{actGene} not in dataset')
            continue

        geneCount = np.zeros([nDigitalSpots,kSpots])
        for spots in enumerate(allSamplesToAllen[actSample]['digitalSpotNearestNeighbors']):
            if np.all(spots[1] < 0):
                geneCount[spots[0]] = 0
            else:
                spotij = np.zeros([7,2], dtype=int)
                spotij[:,1] = np.asarray(spots[1], dtype=int)
                spotij[:,0] = geneIndex
                
                geneCount[spots[0]] = allSamplesToAllen[actSample]['filteredFeatureMatrixMasked'][spotij[:,0],spotij[:,1]]
                
        spotCount = np.nanmean(geneCount, axis=1)
        nTestedSamples += 1
        if experiment['experimental-group'][actSample] == 0:
            digitalSamplesControl[:,startControl:stopControl] = geneCount
            startControl += kSpots
            stopControl += kSpots
            nControls += 1
        elif experiment['experimental-group'][actSample] == 1:
            digitalSamplesExperimental[:,startExperimental:stopExperimental] = geneCount
            startExperimental += kSpots
            stopExperimental += kSpots
            nExperimentals += 1
            
        else:
            continue
    
    
    digitalSamplesControl = np.array(digitalSamplesControl, dtype=float).squeeze()
    digitalSamplesExperimental = np.array(digitalSamplesExperimental, dtype=float).squeeze()

    allTstats = np.zeros(nDigitalSpots)
    allPvals = []
    actTtest = scipy.stats.ttest_ind(digitalSamplesExperimental,digitalSamplesControl, axis=1, nan_policy='propagate')
    actTstats = actTtest[0]
    actPvals = actTtest[1]
    medianDigitalControl = np.nanmedian(digitalSamplesControl,axis=1)
    medianDigitalExperimental = np.nanmedian(digitalSamplesExperimental,axis=1)
    # meanDigitalControl = scipy.stats.mode(digitalSamplesControl,axis=1)
    # meanDigitalExperimental = scipy.stats.mode(digitalSamplesExperimental,axis=1)
    meanDigitalControl = np.nanmean(digitalSamplesControl,axis=1)
    meanDigitalExperimental = np.nanmean(digitalSamplesExperimental,axis=1)

    zeroCenteredCmap = mcolors.TwoSlopeNorm(0,vmin=finiteMin, vmax=finiteMax)
    tTestColormap = zeroCenteredCmap(actTtest[0])
    maxGeneCount = np.nanmax([medianDigitalControl,medianDigitalExperimental])
    # display mean gene count for control group            
    fig = plt.figure()
    fig.add_subplot(1,3,1)
    plt.axis('off')
    plt.imshow(bestSampleToTemplate['visiumTransformed'],cmap='gray')
    plt.scatter(templateDigitalSpots[:,0],templateDigitalSpots[:,1], c=np.array(meanDigitalControl), alpha=0.8, vmin=0,vmax=maxGeneCount,plotnonfinite=False,cmap='Reds',marker='.')
    plt.title('NSD')

    # plt.savefig(os.path.join(derivatives,f'meanGeneCount{actGene}Control.png'), bbox_inches='tight', dpi=300)
    # plt.show()
    # display mean gene count for experimental group
    fig.add_subplot(1,3,2)
    plt.axis('off')
    plt.imshow(bestSampleToTemplate['visiumTransformed'],cmap='gray')
    expScatter = plt.scatter(templateDigitalSpots[:,0],templateDigitalSpots[:,1], c=np.array(meanDigitalExperimental), alpha=0.8, vmin=0,vmax=maxGeneCount,plotnonfinite=False,cmap='Reds',marker='.')
    plt.title('SD')
    fig.colorbar(expScatter,fraction=0.046, pad=0.04)

    fig.add_subplot(1,3,3)
    plt.axis('off')
    plt.imshow(bestSampleToTemplate['visiumTransformed'],cmap='gray')
    tStatScatter = plt.scatter(templateDigitalSpots[:,0],templateDigitalSpots[:,1], c=np.array(actTtest[0]), cmap='seismic',alpha=0.8,vmin=-4,vmax=4,plotnonfinite=False,marker='.')
    plt.title(f't-statistic for {actGene}')
    fig.colorbar(tStatScatter,fraction=0.046, pad=0.04)
    plt.savefig(os.path.join(derivatives,f'tStatGeneCount{actGene}SleepDep.png'), bbox_inches='tight', dpi=300)
    plt.show()
timestr = time.strftime("%Y%m%d-%H%M%S")
print("--- %s seconds ---" % (time.time() - start_time))

