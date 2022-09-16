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
from tkinter import Tk, filedialog, ttk
# import stanlyFunctions
from stanly import importVisiumData, chooseTemplateSlice
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
root.geometry('300x300')

# set directory for derivatives 
# default is set to bids format starting from code folder
outputPath = '../derivatives'
def setOutputDirectory():
    global outputPath
    outputPath = filedialog.askdirectory()

# choose template to use for registration
templateSliceNumber = tkinter.StringVar(value=0)

def setTemplate():
    global templateSliceNumber
    templateWindow = tkinter.Toplevel(root)
    templateWindow.geometry('300x300')
    # need to set max to actual maximum number of images
    templateSpinbox = ttk.Spinbox(templateWindow, textvariable = templateSliceNumber, from_=1, to=100)
    templateSpinbox.pack()
    # print(templateSliceNumber.get())
    # templateSliceNumber = templateSpinbox.get()
    # print(templateSliceNumber)
    # chooseTemplateSlice(templateSliceNumber)
    def selectAndQuit():
        chooseTemplateSlice(int(templateSliceNumber.get()))
        templateWindow.destroy()
    selectTemplateSliceButton = tkinter.Button(templateWindow, text = 'Select this template image?', bd = '5', command = selectAndQuit).pack()


def runSingleRegistration():
    return

def runGroupRegistration():
    return


sampleData = []
def loadSample():   
    samplePath = filedialog.askdirectory()
    global sampleData
    sampleData = importVisiumData(samplePath)
    sampleWindow = tkinter.Toplevel(root)
    sampleImage = ImageTk.PhotoImage(Image.fromarray(np.asarray(rescale(sampleData['imageData'],0.4) * 255)))
    # canvas = tkinter.Canvas(sampleWindow,width=(rescale(sampleData['imageData'],0.4).shape[0] + 20),height=(rescale(sampleData['imageData'],0.4).shape[1] + 20))
    # canvas.grid(row=1,column=1)
    # canvas.create_image(20,20, anchor="nw", image=sampleImage).draw()
    # canvas.draw()
    sampleLabel = tkinter.Label(sampleWindow, image=sampleImage).grid(row=1,column=1)
    processButton = tkinter.Button(sampleWindow, text = 'Process sample?', bd = '5', command = sampleWindow.destroy).grid(row=2,column=1)
    return sampleData

def loadSamplesFromTsv():
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

# create frames of buttons for a cleaner look
nwFrame = tkinter.LabelFrame(root, text="Load Visium samples:").grid(row=0,rowspan=4,column=0, columnspan=2, padx=5, pady=5)
neFrame = tkinter.LabelFrame(root, text="Set output directory:").grid(row=0,rowspan=4, column=1, columnspan=2, padx=5, pady=5)
swFrame = tkinter.LabelFrame(root, text="Registration:").grid(row=3,rowspan=4, column=3, columnspan=2, padx=5, pady=5)
seFrame = tkinter.LabelFrame(root, text="Exit").grid(row=4,rowspan=4, column=3, columnspan=2, padx=5, pady=5)

outputLabel = tkinter.Label(neFrame, text='Output directory').grid(row=1,column=1)
outputEntry = tkinter.Entry(neFrame, textvariable=outputPath, bd = '5').grid(row=2,column=1)
loadSampleButton = tkinter.Button(nwFrame, text = 'Load sample', bd = '5', command = loadSample).grid(row=0,column=0)
# Create buttons for start screen
loadSamplesFromTsvButton = tkinter.Button(nwFrame, text = 'Load samples from .tsv file', bd = '5', command = loadSamplesFromTsv).grid(row=1,column=0)

loadExperimentButton = tkinter.Button(nwFrame, text = 'Load experiment', bd = '5', command = loadExperiment).grid(row=2,column=0)
setDerivativesButton = tkinter.Button(neFrame, text = 'Set output directory', bd = '5', command = setOutputDirectory).grid(row=0,column=1)
setTemplateImageButton = tkinter.Button(swFrame, text = 'Set template image', bd = '5', command = setTemplate).grid(row=1,column=1)
startRegistrationButton = tkinter.Button(swFrame, text = 'Start single image registration', bd = '5', command = runSingleRegistration).grid(row=2,column=1)

quitButton = tkinter.Button(seFrame, text="Quit", bd = '5', command=root.destroy).grid(row=5,column=1)
# currentTemplateLabel = tkinter.Label(root, text=f'Template slice: {templateSliceNumber.get()}').grid(row=5,column=1)
root.mainloop()
