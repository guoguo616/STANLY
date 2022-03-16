#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 16:30:34 2022

@author: zjpeters
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