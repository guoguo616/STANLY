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