#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 16:30:34 2022

none of the following is presumed to be in order, just things that were removed from the extractSpatialInfo
scripts during updates
"""
# this resolution is the mm/pixel estimation of the 6.5mm between dot borders, which is ~1560 pixels in the tissue_hires_image.png
# sampleStartingResolution = 6.5 / 1560
# this resolution is based on each spot being 55um, adjusted to the scale in the scaleFactors setting

#%% extract atlas information
# from allensdk.core.reference_space_cache import ReferenceSpaceCache
# reference_space_key = 'annotation/ccf_2017'
# resolution = 10
# rspc = ReferenceSpaceCache(resolution, reference_space_key, manifest='manifest.json')
# # ID 1 is the adult mouse structure graph
# tree = rspc.get_structure_tree(structure_graph_id=1) 
# regionList = tree.get_name_map()
# hippocampus = tree.get_structures_by_name(['Hippocampal region'])
# hippocampus[0]['id']

# hippocampalMask = np.zeros(templateAnnotationLeft.shape)
# hippocampalMask[templateAnnotationLeft == 1089] = 1

#%% Tissue point registration currently moving into tissue coordinate function
# update tissue points with pre-registration alignment of sample image
# -90 = [[0,1],[-1,0]]
# rotMat = [[0,1],[-1,0]]
# if degreesToRotate == 0:
#     rotMat = [[1,0],[0,1]]
# elif degreesToRotate == 90:
#     rotMat = [[0,-1],[1,0]]
# elif degreesToRotate == 180:
#     rotMat = [[-1,0],[0,-1]]
# elif degreesToRotate == 270:
#     rotMat = [[0,1],[-1,0]]

# # scales tissue coordinates down to image resolution
# tissuePointsResizeToHighRes = sample["tissuePositionsList"][0:, 3:] * sample["scaleFactors"]["tissue_hires_scalef"]
# # below switches x and y in order to properly rotate, this gets undone in next cell
# tissuePointsResizeToHighRes[:,[0,1]] = tissuePointsResizeToHighRes[:,[1,0]]
# plt.imshow(sampleProcessed["tissue"])
# plt.plot(tissuePointsResizeToHighRes[:,0],tissuePointsResizeToHighRes[:,1],marker='.', c='blue', alpha=0.2)
# plt.show()

# tissuePointsResizeRotate = np.matmul(tissuePointsResizeToHighRes, rotMat)
# # below accounts for shift resulting from matrix rotation above, will be different for different angles
# if degreesToRotate == 0:
#     tissuePointsResizeRotate[:,0] = tissuePointsResizeRotate[:,0]
# elif degreesToRotate == 90:
#     tissuePointsResizeRotate[:,1] = tissuePointsResizeRotate[:,1] + sampleProcessed["tissue"].shape[1]
# elif degreesToRotate == 180:
#     tissuePointsResizeRotate[:,0] = tissuePointsResizeRotate[:,0] + sampleProcessed["tissue"].shape[0]
#     tissuePointsResizeRotate[:,1] = tissuePointsResizeRotate[:,1] + sampleProcessed["tissue"].shape[1]
# elif degreesToRotate == 270:
#     tissuePointsResizeRotate[:,0] = tissuePointsResizeRotate[:,0] + sampleProcessed["tissue"].shape[0]
# tissuePointsResizeToTemplate = tissuePointsResizeRotate * resolutionRatio
# plt.imshow(sampleRotate)
# plt.plot(tissuePointsResizeToTemplate[:,0],tissuePointsResizeToTemplate[:,1],marker='.', c='red', alpha=0.2)
# plt.show()

#%% normalize, resize, and rotate image, think about adding to process function
# sampleNorm = cv2.normalize(sampleProcessed['tissue'], None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
# sampleResize = rescale(sampleNorm,resolutionRatio)
# if resize=True not set, image will be slightly misaligned with spots
# sampleRotate = rotate(sampleResize, degreesToRotate, resize=True)
# sampleHistMatch = match_histograms(sampleRotate, template['leftHem'])

#%% run registration of sample to template

# templateAntsImage = ants.from_numpy(template['leftHem'])
# sampleAntsImage = ants.from_numpy(sampleProcessed['tissueHistMatched'])
# # templateAntsImage.set_spacing([templateStartingResolution,templateStartingResolution])
# # sampleAntsImage.set_spacing([templateStartingResolution,templateStartingResolution])

# synXfm = ants.registration(fixed=templateAntsImage, moving=sampleAntsImage, type_of_transform='SyN', grad_step=0.1, reg_iterations=(60,40,20,0), outprefix=os.path.join(sampleProcessed['derivativesPath'],f"{sample['sampleID']}_xfm"))
# ants.plot(templateAntsImage, overlay=synXfm["warpedmovout"])

# # apply syn transform to tissue spot coordinates
# # first line creates a csv file, second line uses that csv as input for antsApplyTransformsToPoints
# np.savetxt(f"{os.path.join(sampleProcessed['derivativesPath'],sample['sampleID'])}_tissuePointsResizeToTemplate.csv",sampleProcessed['tissuePointsForTransform'], delimiter=',', header="x,y,z,t,label,comment")
# os.system(f"antsApplyTransformsToPoints -d 2 -i {os.path.join(sampleProcessed['derivativesPath'],sample['sampleID'])}_tissuePointsResizeToTemplate.csv -o {os.path.join(sampleProcessed['derivativesPath'],sample['sampleID'])}_tissuePointsResizeToTemplateTransformApplied.csv -t [ {os.path.join(sampleProcessed['derivativesPath'],sample['sampleID'])}_xfm0GenericAffine.mat,1] -t {os.path.join(sampleProcessed['derivativesPath'],sample['sampleID'])}_xfm1InverseWarp.nii.gz")

#%% open and check transformed coordinates
# transformedTissuePositionList = []
# with open(os.path.join(f"{os.path.join(sampleProcessed['derivativesPath'],sample['sampleID'])}_tissuePointsResizeToTemplateTransformApplied.csv"), newline='') as csvfile:
#         csvreader = csv.reader(csvfile, delimiter=',')
#         next(csvreader)
#         for row in csvreader:
#             transformedTissuePositionList.append(row)
            
# sampleTransformed = testingXfm["warpedmovout"].numpy()

# transformedTissuePositionList = np.array(transformedTissuePositionList, dtype=float)
# # switching x,y columns back to python compatible and deleting empty columns
# transformedTissuePositionList[:,[0,1]] = transformedTissuePositionList[:,[1,0]]
# transformedTissuePositionList = np.delete(transformedTissuePositionList, [2,3,4,5],1)

# plt.imshow(sampleTransformed)
# plt.scatter(transformedTissuePositionList[0:,0],transformedTissuePositionList[0:,1], marker='.', c='red', alpha=0.3)
# plt.show()

#%% remove any out of bounds points and prepare for comparison to atlas locations

# transformedTissuePositionListMask = np.logical_and(transformedTissuePositionList > 0, transformedTissuePositionList < sampleTransformed.shape[0])

# transformedTissuePositionListFinal = [];
# transformedBarcodesFinal = []
# for i, masked in enumerate(transformedTissuePositionListMask):
#     if masked.all() == True:
#         transformedTissuePositionListFinal.append(transformedTissuePositionList[i])
#         transformedBarcodesFinal.append(sample["tissueSpotBarcodeList"][i])

# transformedTissuePositionListFinal = np.array(transformedTissuePositionListFinal, dtype=float)
#%%
# plt.imshow(template['leftHem'])
# plt.scatter(transformedTissuePositionListFinal[0:,0],transformedTissuePositionListFinal[0:,1], marker='x', c='red', alpha=0.3)
# plt.show()
# plt.imshow

# create a "fake" annotation image by replacing all regions with # > 1500 with one value that just looks better in an overlay
# templateAnnotationLeftFake = template['leftHemAnnot']
# templateAnnotationLeftFake[template['leftHemAnnot'] > 1500] = 100
# plt.imshow(template['leftHemAnnot'])
# plt.scatter(transformedTissuePositionListFinal[0:,0],transformedTissuePositionListFinal[0:,1], marker='.', c='red', alpha=0.5)
# plt.show()
# plt.imshow


# plt.imshow(sampleTransformed)
# plt.scatter(transformedTissuePositionListFinal[0:,0],transformedTissuePositionListFinal[0:,1], marker='x', c='red', alpha=0.3)
# plt.show()
# plt.imshow

# plt.imshow(sampleTransformed,cmap='gray')
# plt.imshow(template['leftHemAnnot'], alpha=0.3)
# plt.show()

#%% reorder the filtered feature matrix to match the spot list
filteredFeatureMatrixString = []
for bytebarcode in sample['filteredFeatureMatrix'][1]:
    filteredFeatureMatrixString.append(bytebarcode.decode())

filteredFeatureMatrixReorder = []
for actbarcode in sampleRegistered['maskedBarcodes']:
    filteredFeatureMatrixReorder.append(filteredFeatureMatrixString.index(actbarcode))

reorderedFilteredFeatureMatrix = sampleProcessed['filteredFeatureMatrixDense'][:,filteredFeatureMatrixReorder]