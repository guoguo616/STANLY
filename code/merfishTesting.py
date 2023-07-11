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
import json
import csv
import cv2
import pandas as pd

#%% location of merfish csv data 
sourcedata = os.path.join('/','home','zjpeters','Documents','visiumalignment','sourcedata','merscopedata')
locOfCellByGeneCsv = os.path.join(sourcedata,'datasets_mouse_brain_map_BrainReceptorShowcase_Slice1_Replicate1_cell_by_gene_S1R1.csv')
locOfCellMetadataCsv = os.path.join(sourcedata,'datasets_mouse_brain_map_BrainReceptorShowcase_Slice1_Replicate1_cell_metadata_S1R1.csv')
# load data as pandas dataframe and extract list of genes
cellByGene = pd.read_csv(locOfCellByGeneCsv)
cellMetadata = pd.read_csv(locOfCellMetadataCsv)
geneList = cellByGene.columns[1:]

#%% display all genes
for actGene in geneList:
    plt.scatter(cellMetadata.center_x,cellMetadata.center_y, c=cellByGene[actGene], marker='.', cmap='Reds')
    plt.title(actGene)
    plt.gca().invert_yaxis()
    plt.show()
