#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 16:23:32 2022

@author: zjpeters
"""
# Spatial Transcriptomic Alignment Non Linearly (STANLy)
# a gui for a series of functions meant for the alignment of and analysis of spatial transcriptomic data
# right now build around visium, but should be applicable to any ST data that associates images and transcriptomic data


# from tkinter import *
import tkinter
from tkinter import Tk, filedialog
# import stanlyFunctions
from stanly import importVisiumData
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib import pyplot as plt
from PIL import ImageTk, Image
import numpy as np
from skimage.transform import rescale
import csv
import os
# open window with set dimensions
root = Tk()
root.title('STANLy')
root.geometry('200x200')
# filedialog.askdirectory()
 
def loadSample():   
    samplePath = filedialog.askdirectory()
    sampleData = importVisiumData(samplePath)
    sampleWindow = tkinter.Toplevel()
    sampleImage = ImageTk.PhotoImage(Image.fromarray(np.asarray(rescale(sampleData['imageData'],0.5) * 255)))
    canvas = tkinter.Canvas(sampleWindow,width=rescale(sampleData['imageData'],0.5).shape[0],height=rescale(sampleData['imageData'],0.5).shape[1])
    canvas.pack()
    canvas.create_image(20,20, anchor="nw", image=sampleImage)
    canvas.draw()
    return sampleData

def loadFromTsv():
    participantsFile = tkinter.filedialog.askopenfilename()
    # sampleList = []
    print(participantsFile)
    with open(os.path.join(participantsFile), newline='') as tsvfile:
        tsvreader = csv.reader(tsvfile, delimiter='\t')
        next(tsvreader)
        sampleListWindow = tkinter.Toplevel()
        t = tkinter.Listbox(sampleListWindow)
        n = 1
        for row in tsvreader:
            t.insert(n, row[0])
            n += 1
            # templateList.append(row[1:])
        t.pack()

def loadExperiment():
    experimentPath = filedialog.askdirectory()
    return experimentPath


loadSampleButton = tkinter.Button(root, text = 'Load sample', bd = '5', command = loadSample).pack()
# Create a Button
loadFromTsvButton = tkinter.Button(root, text = 'Load from .tsv file', bd = '5', command = loadFromTsv).pack()

loadExperimentButton = tkinter.Button(root, text = 'Load experiment', bd = '5', command = loadExperiment).pack()
# loadExperimentButton.pack(side = 'top')
setDerivativesButton = tkinter.Button(root, text = 'Set output directory', bd = '5', command = filedialog.askdirectory).pack()
# loadSampleButton.pack(side = 'bottom')
root.mainloop()
