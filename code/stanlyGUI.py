#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 16:23:32 2022

@author: Zeru Peterson, zeru-peterson@uiowa.edu https://research-git.uiowa.edu/zjpeters/STANLY
"""
# =============================================================================
# Spatial Transcriptomic ANaLYsis (STANLY)
# a gui for a series of functions meant for the alignment of and analysis 
# of spatial transcriptomic data
# =============================================================================

# from tkinter import *
import tkinter
from tkinter import Tk, filedialog, ttk
import sys
# import stanlyFunctions
sys.path.insert(0, "/home/zjpeters/Documents/stanly/code")
import stanly
# from stanly import importVisiumData, chooseTemplateSlice, processVisiumData, rotateTissuePoints, runANTsToAllenRegistration
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib import pyplot as plt
from PIL import ImageTk, Image
import numpy as np
from skimage.transform import rescale
import csv
import os
from glob import glob
# open window with set dimensions
root = Tk()
root.title('STANLY')
root.geometry('500x500')
mainFont = tkinter.font.Font(
    family='Helvetica',
    size='12',
    weight='normal',
    slant='roman',
    underline=0,
    overstrike=0
    )
# set directory for derivatives 
# default is set to bids format starting from code folder
# outputPath = '../derivatives'
def setOutputDirectory():
    global outputPath
    outputPath = filedialog.askdirectory()

templateSlicePath = '../data/allen10umSlices'
templateSliceImages = sorted(glob(os.path.join(templateSlicePath,"*10umSlice*.png")))

templateData = []
# need to consider RH and LH
def chooseTemplate():
    global templateSliceNumber
    global templateData
    global templateLabel
    global actTemplateImage
    templateWindow = tkinter.Toplevel(root)
    templateWindow.title('Select template slice')

    templateSliceNumber = 70
    actTemplateImage = ImageTk.PhotoImage(Image.open(templateSliceImages[templateSliceNumber]))

    templateWindow.geometry(f'{actTemplateImage.width() + 40}x{actTemplateImage.height() + 80}')
    templateLabel = tkinter.Label(templateWindow, text=f'{templateSliceNumber}',image=actTemplateImage)
    templateLabel.place(x=20,y=20)

    def nextClick():
        global templateSliceNumber
        global templateLabel
        
        if templateSliceNumber < 131:
            templateSliceNumber = templateSliceNumber + 1
            actTemplateImage = ImageTk.PhotoImage(Image.open(templateSliceImages[templateSliceNumber]))
        else:
            templateSliceNumber = 0
            actTemplateImage = ImageTk.PhotoImage(Image.open(templateSliceImages[templateSliceNumber]))
        
        templateLabel.config(image=actTemplateImage)
        templateLabel.image = actTemplateImage
        
    def backClick():
        global templateSliceNumber
        global templateLabel
        if templateSliceNumber > -1:
            templateSliceNumber = templateSliceNumber - 1
            actTemplateImage = ImageTk.PhotoImage(Image.open(templateSliceImages[templateSliceNumber]))
            
        else:
            templateSliceNumber = 131
            actTemplateImage = ImageTk.PhotoImage(Image.open(templateSliceImages[templateSliceNumber]))
            
        templateLabel.config(image=actTemplateImage)
        templateLabel.image = actTemplateImage
        
    def selectAndQuit():
        global templateData
        global processButton
        global hemisphere
        hemisphere='whole'
        templateData = stanly.chooseTemplateSlice(templateSliceNumber)
        templateWindow.destroy()
        processButton.config(state=tkinter.NORMAL)
        processButton.state = tkinter.NORMAL
    
    def selectLeftAndQuit():
        global templateData
        global processButton
        global hemisphere
        hemisphere='left'
        templateData = stanly.chooseTemplateSlice(templateSliceNumber)
        templateWindow.destroy()
        processButton.config(state=tkinter.NORMAL)
        processButton.state = tkinter.NORMAL
        
    def selectRightAndQuit():
        global templateData
        global processButton
        global hemisphere
        hemisphere='right'
        templateData = stanly.chooseTemplateSlice(templateSliceNumber)
        templateWindow.destroy()
        processButton.config(state=tkinter.NORMAL)
        processButton.state = tkinter.NORMAL
        
    backButton = tkinter.Button(templateWindow, text = 'Back', bd = '5', command = backClick, font=mainFont)
    backButton.place(x= (actTemplateImage.width()/8),y= (actTemplateImage.height() + 40))
    nextButton = tkinter.Button(templateWindow, text = 'Next', bd = '5', command = nextClick, font=mainFont)
    nextButton.place(x= 7*((actTemplateImage.width())/8),y= (actTemplateImage.height() + 40))
    selectSliceButton = tkinter.Button(templateWindow, text = 'Select whole slice', bd = '5', command = selectAndQuit, font=mainFont)
    selectSliceButton.place(x= (actTemplateImage.width()/2), y = actTemplateImage.height() + 40)
    selectLeftSliceButton = tkinter.Button(templateWindow, text = 'Select left hemisphere', bd = '5', command = selectLeftAndQuit, font=mainFont)
    selectLeftSliceButton.place(x= 2*(actTemplateImage.width()/8), y = actTemplateImage.height() + 40)
    selectRightSliceButton = tkinter.Button(templateWindow, text = 'Select right hemisphere', bd = '5', command = selectLeftAndQuit, font=mainFont)
    selectRightSliceButton.place(x= 6*(actTemplateImage.width()/8), y = actTemplateImage.height() + 40)


    # selectTemplateSliceButton = tkinter.Button(templateWindow, text = 'Select this template image?', bd = '5', command = selectAndQuit)
    # selectTemplateSliceButton.pack()
def rotateClick():
    global sampleImage
    global rotation
    global sampleImageMatrix
    global sampleLabel
    rotation = rotation + 90
    if rotation == 360:
        rotation = 0
    sampleImageMatrix = sampleImageMatrix.rotate(90)
    sampleImage = ImageTk.PhotoImage(sampleImageMatrix)
    sampleLabel.config(image=sampleImage)
    sampleLabel.image = sampleImage

def processClick():
    global sampleImage
    global sampleImageMatrix
    global processedSampleData
    global beginRegistrationButton
    global rotateImageButton
    # global selectTemplateButton
    global registerClick
    global sampleWindow
    try:
        processedSampleData = stanly.processVisiumData(sampleData, templateData, rotation, outputPath)
    except NameError:
        setOutputDirectory()
        processedSampleData = stanly.processVisiumData(sampleData, templateData, rotation, outputPath)
    sampleImageMatrix = Image.fromarray(np.asarray(processedSampleData['tissueProcessed'] * 255))
    sampleImage = ImageTk.PhotoImage(sampleImageMatrix)
    sampleLabel.config(image=sampleImage)
    sampleLabel.image = sampleImage
    rotateImageButton.destroy()
    selectTemplateButton.destroy()
    processButton.destroy()
    # sampleWindow.geometry(f"{processedSampleData['tissueRotated'].shape[0] + 40}x{processedSampleData['tissueRotated'].shape[1] + 80}")
    beginRegistrationButton = tkinter.Button(sampleWindow, text= 'Begin registration?', bd = '5', command = registerClick, font=mainFont)
    beginRegistrationButton.place(x= 2*(sampleImage.width())/4,y= (sampleImage.height() + 40))
    
def resetRotationButtonClick():
    global rotation
    rotation = 0
    

def registerClick():
    global sampleImage
    global sampleImageMatrix
    runSingleRegistration()
    sampleImageMatrix = Image.fromarray(np.asarray(registeredData['tissueRegistered'] * 255))
    sampleImage = ImageTk.PhotoImage(sampleImageMatrix)
    sampleLabel.config(image=sampleImage)
    sampleLabel.image = sampleImage
    beginRegistrationButton.destroy()
    showSpotsButton = tkinter.Button(sampleWindow, text= 'Show spots?', bd = '5', command = showSpots)
    showSpotsButton.place(x= 3*(sampleImage.width())/4,y= (sampleImage.height() + 40))
    resetRotationButton = tkinter.Button(sampleWindow, text='Reset rotation?', bd = '5', command = resetRotationButtonClick, font=mainFont)
    resetRotationButton.place(x = (sampleImage.width())/4,y= (sampleImage.height() + 40))
def showSpots():
    # global sampleWindow
    # figure size is defined in inches
    fig = Figure(figsize = (5, 5), dpi = 150)
    spotPlot = fig.add_subplot(1,1,1)
    spotPlot.scatter(registeredData['transformedTissuePositionList'][0:,0],registeredData['transformedTissuePositionList'][0:,1], marker='.', c='red', alpha=0.3)
    spotPlot.invert_yaxis()
    spotPlot.set_aspect('equal', 'box')
    canvas = FigureCanvasTkAgg(fig, master = sampleWindow)  
    canvas.draw()
    canvas.get_tk_widget().place(x=20,y=20)
    return

def displayGene():
    return
sampleData = []
processedSampleData = []
rotation = 0
# load sample will take as input one folder containing the spatial folder and filtered feature matrix.h5 file
# should probably separate load sample and process so that process can be used by experiment function
def loadSample():   
    global sampleData
    global sampleImage
    global sampleImageMatrix
    global processButton
    global beginRegistrationButton
    global sampleLabel
    global rotateImageButton
    global selectTemplateButton
    global sampleWindow
    samplePath = filedialog.askdirectory()
    sampleWindow = tkinter.Toplevel(root)
    sampleData = stanly.importVisiumData(samplePath)
    sampleImageMatrix = Image.fromarray(np.squeeze(np.asarray(rescale(sampleData['imageData'],0.4) * 255).astype(np.uint8)))
    sampleImage = ImageTk.PhotoImage(sampleImageMatrix)
    h = sampleImage.width() + 40
    w = sampleImage.height() + 80
    sampleWindow.geometry(f'{h}x{w}')
    sampleLabel = tkinter.Label(sampleWindow,image=sampleImage)
    sampleLabel.place(x=20,y=20)
    rotateImageButton = tkinter.Button(sampleWindow, text= 'Rotate 90 degrees', bd = '5', command = rotateClick, font=mainFont)
    rotateImageButton.place(x= (sampleImage.width())/4,y= (sampleImage.height() + 40))
    selectTemplateButton = tkinter.Button(sampleWindow, text = 'Choose template slice', bd = '5', command = chooseTemplate, font=mainFont)
    selectTemplateButton.place(x= 2*((sampleImage.width())/4),y= (sampleImage.height() + 40))
    processButton = tkinter.Button(sampleWindow, text = 'Process sample?', bd = '5', command = processClick, state=tkinter.DISABLED, font=mainFont)
    processButton.place(x= 3*((sampleImage.width())/4),y= (sampleImage.height() + 40))

    return processedSampleData

# def loadSamplesFromTsv():
#     participantsFile = tkinter.filedialog.askopenfilename()
#     # sampleList = []
#     print(participantsFile)
#     with open(os.path.join(participantsFile), newline='') as tsvfile:
#         tsvreader = csv.reader(tsvfile, delimiter='\t')
#         next(tsvreader)
#         sampleListWindow = tkinter.Toplevel()
#         t = tkinter.Listbox(sampleListWindow)
#         n = 1
#         for row in tsvreader:
#             t.insert(n, row[0])
#             n += 1
#             # templateList.append(row[1:])
#         t.pack()
# maybe change from loadExperiment to loadMultipleSamples
# load experiment will take as input one folder containing multiple sample folders as described for load sample
sampleList = []
rotationList = []
experimentalGroupList = []
# experiment = {'sample-id': sampleList,
              # 'rotation': rotationList,
              # 'experimental-group': experimentalGroupList}


def addToExperiment():
    sampleList.append()
    return
# actSampleData = []
nOfSamples = 0
def loadExperiment():
    global sampleData
    global sampleImage
    global sampleImageMatrix
    global processButton
    global beginRegistrationButton
    global sampleLabel
    global rotateImageButton
    global selectTemplateButton
    global sampleWindow
    global rotation
    sampleWindow = tkinter.Toplevel(root)
    experimentPath = filedialog.askdirectory()
    for sampleDir in os.listdir(experimentPath):
        if os.path.isdir(os.path.join(experimentPath, sampleDir)):
            rotation = 0
            sampleData = stanly.importVisiumData(os.path.join(experimentPath, sampleDir))
            sampleImageMatrix = Image.fromarray(np.asarray(rescale(sampleData['imageData'],0.4) * 255))
            sampleImage = ImageTk.PhotoImage(sampleImageMatrix)
            h = sampleImage.width() + 40
            w = sampleImage.height() + 80
            sampleWindow.geometry(f'{h}x{w}')
            sampleLabel = tkinter.Label(sampleWindow,image=sampleImage)
            sampleLabel.place(x=20,y=20)
            rotateImageButton = tkinter.Button(sampleWindow, text= 'Rotate 90 degrees', bd = '5', command = rotateClick, font=mainFont)
            rotateImageButton.place(x= (sampleImage.width())/4,y= (sampleImage.height() + 40))
            selectTemplateButton = tkinter.Button(sampleWindow, text = 'Choose template slice', bd = '5', command = chooseTemplate, font=mainFont)
            selectTemplateButton.place(x= 2*((sampleImage.width())/4),y= (sampleImage.height() + 40))
            processButton = tkinter.Button(sampleWindow, text = 'Process sample?', bd = '5', command = processClick, state=tkinter.DISABLED, font=mainFont)
            processButton.place(x= 3*((sampleImage.width())/4),y= (sampleImage.height() + 40))

    return experimentPath

registeredData = []
def runSingleRegistration():
    global processedSampleData
    global registeredData

    registeredData = stanly.runANTsToAllenRegistration(processedSampleData, templateData)

    return

def runGroupRegistration():
    return

# outputLabel = tkinter.Label(root, text='Output directory').grid(row=0, column=2)
# outputEntry = tkinter.Entry(root, textvariable=outputPath, bd = '5').grid(row=1,column=2)
loadSampleButton = tkinter.Button(root, text = 'Load sample', bd = '5', command = loadSample, font=mainFont).grid(row=0,column=0)
# Create buttons for start screen
# loadSamplesFromTsvButton = tkinter.Button(root, text = 'Load samples from .tsv file', bd = '5', command = loadSamplesFromTsv).grid(row=1,column=0)

loadExperimentButton = tkinter.Button(root, text = 'Load experiment', bd = '5', command = loadExperiment, font=mainFont).grid(row=1,column=0)
setDerivativesButton = tkinter.Button(root, text = 'Set output directory', bd = '5', command = setOutputDirectory, font=mainFont).grid(row=0, column=1)
setTemplateImageButton = tkinter.Button(root, text = 'Set template image', bd = '5', command = chooseTemplate, font=mainFont).grid(row=3, column=0)
startRegistrationButton = tkinter.Button(root, text = 'Start single image registration', bd = '5', command = runSingleRegistration, font=mainFont).grid(row=1, column=1)

quitButton = tkinter.Button(root, text="Quit", bd = '5', command=root.destroy, font=mainFont).grid(row=3, column=1)
root.mainloop()
