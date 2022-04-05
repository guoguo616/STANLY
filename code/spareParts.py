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


#%% #################################################################################
# below was taken out on 3/25 in order to start working on intersample registration #
#####################################################################################

#%% calculate pairwise distance for each points in a sample
# kNN here is how many nearest neighbors we want to calculate
kNN = 36

pairwiseSquareMatrix = {}
pairwiseNearestNeighbors = {}
nearestNeighborEdges = {}
####
# need to adjust/build edges, since right now two nearest neighbors with the
# same distance is causing a crash because of multiple indices
#### ^ was a euclidean metric issue, changing metric in pdist fixes
for actSample in range(len(experimentalResults)):    
    print(truncExperiment['sample-id'][actSample])
    samplePDist = []
    samplePDist = pdist(experimentalResults[actSample]['maskedTissuePositionList'], metric='cosine')
    samplePDistSM = []
    samplePDistSM = squareform(samplePDist)
    pairwiseSquareMatrix[actSample] = samplePDistSM
    samplePDistSMSorted = []
    samplePDistSMSorted = np.sort(samplePDistSM, axis=1)
    # below contains kNN distances for each in tissue spot based on post alignment distance
    # samplePDistNN = []
    # samplePDistNN = samplePDistSMSorted[:,1:kNN+1]
    samplePDistEdges = []
    # output of samplekNN should contain the barcode indices of all of the nearest neighbors
    samplekNN = np.zeros([samplePDistSM.shape[0],kNN])
    for i, row in enumerate(samplePDistSM):
        samplePDistNN = []
        # samplePDistNN = samplePDistSMSorted[i,1:kNN+1]
        if samplePDistSMSorted[i,1] > 0:
            samplePDistNN = samplePDistSMSorted[i,1:kNN+1]
            for sigK in range(kNN):
                samplekNN[i,sigK] = np.argwhere(row == samplePDistNN[sigK])
                samplePDistEdges.append([i,np.argwhere(row == samplePDistNN[sigK])]) 
        else:
            samplePDistNN = samplePDistSMSorted[i,2:kNN+2]
            for sigK in range(kNN):
                samplekNN[i,sigK] = np.argwhere(row == samplePDistNN[sigK])
                samplePDistEdges.append([i,np.argwhere(row == samplePDistNN[sigK])]) 
                # samplePDistEdges[1,i] = 
            
    pairwiseNearestNeighbors[actSample] = samplekNN
    nearestNeighborEdges[actSample] = samplePDistEdges
#%% take nearest neighbor lists and turn into list of coordinate edges i.e. [I,J] 


allEdges = {}

# goes through each sample and creates a list of edges defined by two points [i,j]
for i in range(len(pairwiseNearestNeighbors)):
    actSample = pairwiseNearestNeighbors[i]
    sampleEdges = np.empty((0,2), int)
    for j, row in enumerate(actSample):
        # print(row)
        actEdges = np.zeros([kNN,2])
        actEdges[:,0] = j
        actEdges[:,1] = np.transpose(row)
        sampleEdges = np.append(sampleEdges, actEdges, axis=0)
        
    # removes any duplicate edges (i.e. edges [i,j] and [j,i] are identical)
    # sortedEdges = np.sort(sampleEdges, axis=1)
    # uniqueEdges = np.array(np.unique(sortedEdges, axis=0),dtype=int)
    allEdges[i] = np.array(sampleEdges,dtype=int)



#%% run cosine similarity on kNN spots
# can now use edge lists to create weighted adjacency matrices

# actSample = 0
actEdges = allEdges[0]
# sample = importVisiumData(os.path.join(rawdata, experiment['sample-id'][actSample]))
# template = chooseTemplateSlice(experiment['template-slice'][actSample])
# sampleProcessed = processVisiumData(sample, template, experiment['rotation'][actSample])
# sampleRegistered = runANTsRegistration(sampleProcessed, template)
V = []
cosineSim = []
# weighted adjacency matrix
W = np.zeros([actEdges.max()+1,actEdges.max()+1])
for row in actEdges:
    V = cosine(experimentalResults[0]['filteredFeatureMatrixMasked'][:,row[0]], experimentalResults[0]['filteredFeatureMatrixMasked'][:,row[1]])
    W[row[0],row[1]] = V
    W[row[1],row[0]] = V
    
# calculate cosine sim for all options, not just connected
# for row in range(experimentalResults[0]['filteredFeatureMatrixMasked'].shape[1]):
#     for col in range(experimentalResults[0]['filteredFeatureMatrixMasked'].shape[1]):
#         V = cosine(experimentalResults[0]['filteredFeatureMatrixMasked'][:,row], experimentalResults[0]['filteredFeatureMatrixMasked'][:,col])
#         W[row,col] = 1 - V
# weighted laplacian
D = sum(W)

D = np.diag(D)

L = D - W

Lsparse = sp_sparse.csc_matrix(L)

# spectral embedding

sampleEigVal, sampleEigVec = sp_sparse.linalg.eigs(Lsparse, k=6)

# sampleEigVecSorted = np.sort(sampleEigVec)
# sampleEigVecSortedIdx = np.argsort(sampleEigVec)
# from sklearn import manifold
# se = manifold.SpectralEmbedding(n_components=2, n_neighbors=kNN)
# seData = se.fit_transform(L).T


#%% next steps towards clustering data, though this could realistically be replaced by something like BayesSpace
from sklearn.cluster import AffinityPropagation, KMeans, SpectralClustering
# from sklearn import metrics
afprop = AffinityPropagation(max_iter=250, affinity='precomputed')
afprop.fit(np.real(L))
cluster_centers_indices = afprop.cluster_centers_indices_
n_clusters_ = len(cluster_centers_indices)
# Predict the cluster for all the samples
# P = afprop.predict(np.real(L))

km = KMeans(n_clusters=36).fit(np.real(L))

# se = SpectralClustering(n_clusters=kNN, assign_labels='discretize', affinity='precomputed').fit(L)
# can now plot clusters onto spots using P as color option
plt.imshow(experimentalResults[0]['visiumTransformed'],cmap='gray')
plt.scatter(experimentalResults[0]['maskedTissuePositionList'][0:,0],experimentalResults[0]['maskedTissuePositionList'][0:,1], marker='.', c=afprop.labels_, alpha=0.3)

plt.show()
# from scipy import spatial

#%% create template from intersample registration
allSampleImage = np.zeros(experimentalResults[0]['visiumTransformed'].shape)
for i in range(len(experimentalResults)):
    allSampleImage = allSampleImage + experimentalResults[i]['visiumTransformed']
    
meanSample = allSampleImage / (len(experimentalResults) + 1)
    
