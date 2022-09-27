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
from stanly import importVisiumData, chooseTemplateSlice, processVisiumData, rotateTissuePoints, runANTsToAllenRegistration
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
root.title('STANLy')
root.geometry('500x500')

# set directory for derivatives 
# default is set to bids format starting from code folder
outputPath = '../derivatives'
def setOutputDirectory():
    global outputPath
    outputPath = filedialog.askdirectory()


templateSlicePath = '../data/allen10umSlices'
templateLeftSliceImages = sorted(glob(os.path.join(templateSlicePath,"*LH*.png")))

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
    actTemplateImage = ImageTk.PhotoImage(Image.open(templateLeftSliceImages[templateSliceNumber]))

    templateWindow.geometry(f'{actTemplateImage.width() + 40}x{actTemplateImage.height() + 80}')
    templateLabel = tkinter.Label(templateWindow, text=f'{templateSliceNumber}',image=actTemplateImage)
    templateLabel.place(x=20,y=20)

    def nextClick():
        global templateSliceNumber
        global templateLabel
        
        if templateSliceNumber < 131:
            templateSliceNumber = templateSliceNumber + 1
            actTemplateImage = ImageTk.PhotoImage(Image.open(templateLeftSliceImages[templateSliceNumber]))
        else:
            templateSliceNumber = 0
            actTemplateImage = ImageTk.PhotoImage(Image.open(templateLeftSliceImages[templateSliceNumber]))
        
        templateLabel.config(image=actTemplateImage)
        templateLabel.image = actTemplateImage
        
    def backClick():
        global templateSliceNumber
        global templateLabel
        if templateSliceNumber > -1:
            templateSliceNumber = templateSliceNumber - 1
            actTemplateImage = ImageTk.PhotoImage(Image.open(templateLeftSliceImages[templateSliceNumber]))
            
        else:
            templateSliceNumber = 131
            actTemplateImage = ImageTk.PhotoImage(Image.open(templateLeftSliceImages[templateSliceNumber]))
            
        templateLabel.config(image=actTemplateImage)
        templateLabel.image = actTemplateImage
        
    def selectAndQuit():
        global templateData
        global processButton
        templateData = chooseTemplateSlice(templateSliceNumber)
        templateWindow.destroy()
        processButton.config(state=tkinter.NORMAL)
        processButton.state = tkinter.NORMAL
        
    backButton = tkinter.Button(templateWindow, text = 'Back', bd = '5', command = backClick)
    backButton.place(x= (actTemplateImage.width())/4,y= (actTemplateImage.height() + 40))
    nextButton = tkinter.Button(templateWindow, text = 'Next', bd = '5', command = nextClick)
    nextButton.place(x= 3*((actTemplateImage.width())/4),y= (actTemplateImage.height() + 40))
    selectSliceButton = tkinter.Button(templateWindow, text = 'Select slice', bd = '5', command = selectAndQuit)
    selectSliceButton.place(x= (actTemplateImage.width())/2, y = actTemplateImage.height() + 40)

    # selectTemplateSliceButton = tkinter.Button(templateWindow, text = 'Select this template image?', bd = '5', command = selectAndQuit)
    # selectTemplateSliceButton.pack()

# thoughts for how to incorporate template selection
'''
can select images from every 10 slices of the ara_data and output as pngs
let the user scroll through these images and select the best fit
can also select the rotation needed to fit
'''
sampleData = []
processedSampleData = []
rotation = 0
def loadSample():   
    samplePath = filedialog.askdirectory()
    global sampleData
    global sampleImage
    global sampleImageMatrix
    global processButton
    global beginRegistrationButton
    sampleData = importVisiumData(samplePath)
    sampleImageMatrix = Image.fromarray(np.asarray(rescale(sampleData['imageData'],0.4) * 255))
    sampleImage = ImageTk.PhotoImage(sampleImageMatrix)
    sampleWindow = tkinter.Toplevel(root)
    h = sampleImage.width() + 40
    w = sampleImage.height() + 80
    sampleWindow.geometry(f'{h}x{w}')
    # canvas = tkinter.Canvas(sampleWindow, width = sampleImage.width(), height = sampleImage.height())      
    # canvas.place(x=0,y=0)
    # canvas.create_image(20,20, image=sampleImage,anchor="nw")
    sampleLabel = tkinter.Label(sampleWindow,image=sampleImage)
    sampleLabel.place(x=20,y=20)
    # def processSample():
    #     global processedSampleData
    #     processedSampleData = processVisiumData(sampleData, templateData, rotation)
    def rotateClick():
        global sampleImage
        global rotation
        global sampleImageMatrix
        if rotation < 360:
            rotation = rotation + 90
        else:
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
        processedSampleData = processVisiumData(sampleData, templateData, rotation)
        sampleImageMatrix = Image.fromarray(np.asarray(processedSampleData['tissueRotated'] * 255))
        sampleImage = ImageTk.PhotoImage(sampleImageMatrix)
        sampleLabel.config(image=sampleImage)
        sampleLabel.image = sampleImage
        rotateImageButton.destroy()
        selectTemplateButton.destroy()
        processButton.destroy()
        # sampleWindow.geometry(f"{processedSampleData['tissueRotated'].shape[0] + 40}x{processedSampleData['tissueRotated'].shape[1] + 80}")
        beginRegistrationButton = tkinter.Button(sampleWindow, text= 'Begin registration?', bd = '5', command = registerClick)
        beginRegistrationButton.place(x= 2*(sampleImage.width())/4,y= (sampleImage.height() + 40))

        # sampleWindow.destroy()
    def registerClick():
        global sampleImage
        global sampleImageMatrix
        runSingleRegistration()
        sampleImageMatrix = Image.fromarray(np.asarray(registeredData['visiumTransformed'] * 255))
        sampleImage = ImageTk.PhotoImage(sampleImageMatrix)
        sampleLabel.config(image=sampleImage)
        sampleLabel.image = sampleImage
        beginRegistrationButton.destroy()

    rotateImageButton = tkinter.Button(sampleWindow, text= 'Rotate 90 degrees', bd = '5', command = rotateClick)
    rotateImageButton.place(x= (sampleImage.width())/4,y= (sampleImage.height() + 40))
    selectTemplateButton = tkinter.Button(sampleWindow, text = 'Choose template slice', bd = '5', command = chooseTemplate)
    selectTemplateButton.place(x= 2*((sampleImage.width())/4),y= (sampleImage.height() + 40))
    processButton = tkinter.Button(sampleWindow, text = 'Process sample?', bd = '5', command = processClick, state=tkinter.DISABLED)
    processButton.place(x= 3*((sampleImage.width())/4),y= (sampleImage.height() + 40))
    return processedSampleData

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

registeredData = []
def runSingleRegistration():
    global processedSampleData
    global registeredData

    registeredData = runANTsToAllenRegistration(processedSampleData, templateData)

    return

def runGroupRegistration():
    return

# outputLabel = tkinter.Label(root, text='Output directory').grid(row=0, column=2)
# outputEntry = tkinter.Entry(root, textvariable=outputPath, bd = '5').grid(row=1,column=2)
loadSampleButton = tkinter.Button(root, text = 'Load sample', bd = '5', command = loadSample).grid(row=0,column=0)
# Create buttons for start screen
# loadSamplesFromTsvButton = tkinter.Button(root, text = 'Load samples from .tsv file', bd = '5', command = loadSamplesFromTsv).grid(row=1,column=0)

loadExperimentButton = tkinter.Button(root, text = 'Load experiment', bd = '5', command = loadExperiment).grid(row=1,column=0)
setDerivativesButton = tkinter.Button(root, text = 'Set output directory', bd = '5', command = setOutputDirectory).grid(row=0, column=1)
setTemplateImageButton = tkinter.Button(root, text = 'Set template image', bd = '5', command = chooseTemplate).grid(row=3, column=0)
startRegistrationButton = tkinter.Button(root, text = 'Start single image registration', bd = '5', command = runSingleRegistration).grid(row=1, column=1)

quitButton = tkinter.Button(root, text="Quit", bd = '5', command=root.destroy).grid(row=3, column=1)
root.mainloop()
