#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 10:53:43 2022

@author: zjpeters
"""
#%% extract atlas information
from allensdk.core.reference_space_cache import ReferenceSpaceCache
reference_space_key = 'annotation/ccf_2017'
resolution = 10
rspc = ReferenceSpaceCache(resolution, reference_space_key, manifest='manifest.json')
rsp = rspc.get_reference_space()
#%%
# ID 1 is the adult mouse structure graph
tree = rspc.get_structure_tree(structure_graph_id=1) 
regionList = tree.get_name_map()
hippocampus = tree.get_structures_by_name(['Hippocampal formation'])

hippocampal3dMask = rsp.make_structure_mask([hippocampus[0]['id']])
hippocampalMask = hippocampal3dMask[700,:,570:]

rhinalFissure = tree.get_structures_by_name(['rhinal fissure'])
rhinalFissure3dMask = rsp.make_structure_mask([rhinalFissure[0]['id']])
rhinalFissureMask = rhinalFissure3dMask[700,:,570:]

bestTemplateSlice10 = bestTemplateSlice * 10
plt.imshow(bestSampleToTemplate['visiumTransformed'])
plt.imshow(hippocampalMask, alpha=0.3)
plt.show()

plt.imshow(rhinalFissureMask)
plt.show()
bestTemplateSlice10 = bestTemplateSlice * 10
plt.imshow(bestSampleToTemplate['visiumTransformed'])
plt.imshow(hippocampalMask, alpha=0.3)
plt.show()
#%% create digital hippocampal spots

hippocampalDigitalTemplateSpots = []
# spotIdx gives a list of spots within the hippocampal formation
spotIdx = []
for row in range(len(roundedTemplateSpots)):
    # 15 in the following is just to erode around the edge of the brain
    if hippocampalMask[roundedTemplateSpots[row,1],roundedTemplateSpots[row,0]] == 1:
        hippocampalDigitalTemplateSpots.append(templateSpots[row])
        spotIdx.append(row)
        
hippocampalDigitalTemplateSpots = np.array(hippocampalDigitalTemplateSpots)
plt.imshow(template['leftHem'])
plt.scatter(hippocampalDigitalTemplateSpots[:,0],hippocampalDigitalTemplateSpots[:,1], alpha=0.3)
plt.show()

#%% check for significant genes within a region/list of regions
# needs to take an input of regions defined in allen ccf
# should be able to use original spots when searching roi
# run through entire gene list looking for a change in expression between conditions
nSigGenes = 0
partialGeneList = list(allSampleGeneList)[0:500]
for nOfGenesChecked,actGene in enumerate(partialGeneList):
    digitalSamplesControl = []
    digitalSamplesExperimental = []
    nTestedSamples = 0
    nControls = 0
    nExperimentals = 0
    for actSample in range(nTotalSamples):
        try:
            geneIndex = allSamplesToAllen[actSample]['geneListMasked'].index(actGene)
            spotCheck = np.count_nonzero(allSamplesToAllen[actSample]['filteredFeatureMatrixMasked'][spotIdx,:])
            # need to do better than "no expression" because that eliminates the fact that the amount is actually 0, not untested
            
            # use binary mask to remove any tissue spots with no expression
            if spotCheck < 15:
                continue
            actNN, actCDist = findDigitalNearestNeighbors(hippocampalDigitalTemplateSpots, allSamplesToAllen[actSample]['maskedTissuePositionList'], kSpots)
            geneCount = np.zeros([hippocampalDigitalTemplateSpots.shape[0],kSpots])
            for spots in enumerate(actNN):
                if ~np.all(spots[1]):
                    geneCount[spots[0]] = 0
                else:
                    geneCount[spots[0]] = allSamplesToAllen[actSample]['filteredFeatureMatrixMasked'][geneIndex,actNN[spots[0]]]
                    
            geneCount = geneCount.reshape([-1])
            nTestedSamples += 1
            if truncExperiment['experimental-group'][actSample] == 0:
                digitalSamplesControl.append(geneCount)
                nControls += 1
            elif truncExperiment['experimental-group'][actSample] == 1:
                digitalSamplesExperimental.append(geneCount)
                nExperimentals += 1
                
            else:
                continue
            
        except:
            continue
        
    if spotCheck < 15:
        continue    
    digitalSamplesControl = np.array(digitalSamplesControl)
    digitalSamplesExperimental = np.array(digitalSamplesExperimental)
    maskedTtests = []
    allTstats = []
    allPvals = []
    if ~digitalSamplesControl.any() or ~digitalSamplesExperimental.any():
        continue
    else:    
        actTtest = scipy.stats.ttest_ind(digitalSamplesExperimental,digitalSamplesControl, nan_policy='omit', axis=None)
        maskedDigitalCoordinates = []
        maskedMeanDigitalControls = []
        maskedMeanDigitalExperimentals = []
        meanDigitalControlHippocampalFormation = np.mean(digitalSamplesControl)
        meanDigitalExperimentalHippocampalFormation = np.mean(digitalSamplesExperimental)
        if actTtest[1] <= 0.05:
            nSigGenes += 1
            maxGeneCount = np.max([meanDigitalControls,meanDigitalExperimentals])
            tStatColor = np.full(hippocampalDigitalTemplateSpots.shape[0],actTtest[0])
            plt.imshow(bestSampleToTemplate['visiumTransformed'])
            if actTtest[0] < 0:
                plt.scatter(hippocampalDigitalTemplateSpots[:,0],hippocampalDigitalTemplateSpots[:,1], c='blue', alpha=0.8,plotnonfinite=False)
            else:
                plt.scatter(hippocampalDigitalTemplateSpots[:,0],hippocampalDigitalTemplateSpots[:,1], c='red', alpha=0.8,plotnonfinite=False)

            # plt.scatter(hippocampalDigitalTemplateSpots[:,0],hippocampalDigitalTemplateSpots[:,1], c=tStatColor, alpha=0.8, vmin=0,vmax=maxGeneCount,plotnonfinite=False)
            # plt.title(f'Mean gene count for {actGene}, control')
            plt.title(f'{actGene}, non sleep deprived, hippocampal expression p <= 0.05')
            plt.colorbar()
            plt.show()
            
        else:
            continue
        
        
        
print("--- %s seconds ---" % (time.time() - start_time))
