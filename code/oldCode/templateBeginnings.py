#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 08:42:51 2022

@author: zjpeters
"""
def runANTsRegistrationToBestFit(processedVisium, bestSliceProcessed):
    # convert into ants image type
    registeredData = {}
    templateAntsImage = ants.from_numpy(bestSliceProcessed['tissueHistMatched'])
    sampleAntsImage = ants.from_numpy(processedVisium['tissueHistMatched'])
    synXfm = ants.registration(fixed=templateAntsImage, moving=sampleAntsImage, \
    type_of_transform='SyNAggro', grad_step=0.1, reg_iterations=(100,80,60,40,20,0), \
    syn_sampling=2, flow_sigma=2, syn_metric='mattes', outprefix=os.path.join(processedVisium['derivativesPath'],f"{processedVisium['sampleID']}_xfm"))
    registeredData['antsOutput'] = synXfm
    ants.plot(templateAntsImage, overlay=synXfm["warpedmovout"])
    # apply syn transform to tissue spot coordinates
    # first line creates a csv file, second line uses that csv as input for antsApplyTransformsToPoints
    np.savetxt(f"{os.path.join(processedVisium['derivativesPath'],processedVisium['sampleID'])}_tissuePointsResizeToTemplate.csv",processedVisium['tissuePointsForTransform'], delimiter=',', header="x,y,z,t,label,comment")
    os.system(f"antsApplyTransformsToPoints -d 2 -i {os.path.join(processedVisium['derivativesPath'],processedVisium['sampleID'])}_tissuePointsResizeToTemplate.csv -o {os.path.join(processedVisium['derivativesPath'],processedVisium['sampleID'])}_tissuePointsResizeToTemplateTransformApplied.csv -t [ {os.path.join(processedVisium['derivativesPath'],processedVisium['sampleID'])}_xfm0GenericAffine.mat,1] -t {os.path.join(processedVisium['derivativesPath'],processedVisium['sampleID'])}_xfm1InverseWarp.nii.gz")
    
    transformedTissuePositionList = []
    with open(os.path.join(f"{os.path.join(processedVisium['derivativesPath'],processedVisium['sampleID'])}_tissuePointsResizeToTemplateTransformApplied.csv"), newline='') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',')
            next(csvreader)
            for row in csvreader:
                transformedTissuePositionList.append(row)
                
    registeredData['visiumTransformed'] = synXfm["warpedmovout"].numpy()

    registeredData['transformedTissuePositionList'] = np.array(transformedTissuePositionList, dtype=float)
    # switching x,y columns back to python compatible and deleting empty columns
    registeredData['transformedTissuePositionList'][:,[0,1]] = registeredData['transformedTissuePositionList'][:,[1,0]]
    registeredData['transformedTissuePositionList'] = np.delete(registeredData['transformedTissuePositionList'], [2,3,4,5],1)

    plt.imshow(registeredData['visiumTransformed'])
    plt.scatter(registeredData['transformedTissuePositionList'][0:,0],registeredData['transformedTissuePositionList'][0:,1], marker='.', c='red', alpha=0.3)
    plt.show()
    
    plt.imshow(registeredData['visiumTransformed'],cmap='gray')
    plt.imshow(bestSliceProcessed['tissueHistMatched'], alpha=0.3)
    plt.title(processedVisium['sampleID'])
    plt.show()
        
    transformedTissuePositionListMask = np.logical_and(registeredData['transformedTissuePositionList'] > 0, registeredData['transformedTissuePositionList'] < registeredData['visiumTransformed'].shape[0])
    
    transformedTissuePositionListFinal = [];
    transformedBarcodesFinal = []
    for i, masked in enumerate(transformedTissuePositionListMask):
        if masked.all() == True:
            transformedTissuePositionListFinal.append(registeredData['transformedTissuePositionList'][i])
            transformedBarcodesFinal.append(processedVisium["tissueSpotBarcodeList"][i])
    
    registeredData['maskedTissuePositionList'] = np.array(transformedTissuePositionListFinal, dtype=float)
    registeredData['maskedBarcodes'] = transformedBarcodesFinal
    return registeredData


processedSamples = {}
for actSample in range(len(truncExperiment['sample-id'])):
    sample = importVisiumData(os.path.join(rawdata, truncExperiment['sample-id'][actSample]))
    template = chooseTemplateSlice(truncExperiment['template-slice'][actSample])
    sampleProcessed = processVisiumData(sample, template, truncExperiment['rotation'][actSample])
    processedSamples[actSample] = sampleProcessed

for actSample in range(len(processedSamples)):
    sampleRegistered = runANTsRegistrationToBestFit(processedSamples[actSample], processedSamples[4])
    experimentalResults[actSample] = sampleRegistered
