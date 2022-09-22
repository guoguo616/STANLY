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
from stanly import importVisiumData, chooseTemplateSlice, processVisiumData
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
# templateWindow.geometry('300x300')
# choose template to use for registration
# templateSliceNumber = tkinter.StringVar(value=0)
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

    # templateWindow = tkinter.Toplevel(root)
    # templateWindow.geometry('300x300')
    # templateWindow.title('Select template slice')
    
    # need to set max to actual maximum number of images
    # templateSpinbox = ttk.Spinbox(templateWindow, textvariable = templateSliceNumber, from_=1, to=131)
    # templateSpinbox.pack()
    
    # actTemplateImage = ImageTk.PhotoImage(Image.open(templateLeftSliceImages[templateSliceNumber]))
    # h = actTemplateImage.width() + 40
    # w = actTemplateImage.height() + 80
    # templateWindow.geometry(f'{h}x{w}')
    # templateFrame = tkinter.Frame(templateWindow, width=250, height=250, bg='white').place(x=0,y=0)
    
    # templateLabel = tkinter.Label(templateWindow, image=actTemplateImage).place(x=0,y=0)
    # templateLabel.config
    # canvas = tkinter.Canvas(templateWindow, width = templateWindow.width(), height = templateWindow.height())      
    # canvas.place(x=0,y=0)
    # canvas.create_image(20,20, image=actTemplateImage,anchor="nw")
    # def processSample():
    #     global processedSampleData
    #     processedSampleData = processVisiumData(sampleData, templateData, rotation)
    # sampleData = processVisiumData()
    # would be good to include calculated paxinos printed on window
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
        templateData = chooseTemplateSlice(templateSliceNumber)
        templateWindow.destroy()
        
    backButton = tkinter.Button(templateWindow, text = 'Back', bd = '5', command = backClick)
    backButton.place(x= (actTemplateImage.width() + 40)/4,y= (actTemplateImage.height() + 40))
    nextButton = tkinter.Button(templateWindow, text = 'Next', bd = '5', command = nextClick)
    nextButton.place(x= 3*((actTemplateImage.width() + 40)/4),y= (actTemplateImage.height() + 40))
    selectSliceButton = tkinter.Button(templateWindow, text = 'Select slice', bd = '5', command = selectAndQuit)
    selectSliceButton.place(x= (actTemplateImage.width() + 40)/2, y = actTemplateImage.height() + 40)

    # selectTemplateSliceButton = tkinter.Button(templateWindow, text = 'Select this template image?', bd = '5', command = selectAndQuit)
    # selectTemplateSliceButton.pack()

# thoughts for how to incorporate template selection
'''
can select images from every 10 slices of the ara_data and output as pngs
let the user scroll through these images and select the best fit
can also select the rotation needed to fit
'''
def runSingleRegistration():
    return

def runGroupRegistration():
    return

# add rotate option to load sample window

sampleData = []
processedSampleData = []
def loadSample():   
    samplePath = filedialog.askdirectory()
    global sampleData
    global sampleImage
    
    sampleData = importVisiumData(samplePath)
    sampleImage = ImageTk.PhotoImage(Image.fromarray(np.asarray(rescale(sampleData['imageData'],0.4) * 255)))
    sampleWindow = tkinter.Toplevel(root)
    h = sampleImage.width() + 40
    w = sampleImage.height() + 80
    sampleWindow.geometry(f'{h}x{w}')
    canvas = tkinter.Canvas(sampleWindow, width = sampleImage.width(), height = sampleImage.height())      
    canvas.place(x=0,y=0)
    canvas.create_image(20,20, image=sampleImage,anchor="nw")
    # def processSample():
    #     global processedSampleData
    #     processedSampleData = processVisiumData(sampleData, templateData, rotation)
    sampleData = processVisiumData()
    processButton = tkinter.Button(sampleWindow, text = 'Process sample?', bd = '5', command = sampleWindow.destroy).place(x= (sampleImage.width() + 40)/2,y= (sampleImage.height() + 40))
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

outputLabel = tkinter.Label(neFrame, text='Output directory').grid(row=0, column=2)
outputEntry = tkinter.Entry(neFrame, textvariable=outputPath, bd = '5').grid(row=1,column=2)
loadSampleButton = tkinter.Button(nwFrame, text = 'Load sample', bd = '5', command = loadSample).grid(row=0,column=0)
# Create buttons for start screen
loadSamplesFromTsvButton = tkinter.Button(nwFrame, text = 'Load samples from .tsv file', bd = '5', command = loadSamplesFromTsv).grid(row=1,column=0)

loadExperimentButton = tkinter.Button(nwFrame, text = 'Load experiment', bd = '5', command = loadExperiment).grid(row=2,column=0)
setDerivativesButton = tkinter.Button(neFrame, text = 'Set output directory', bd = '5', command = setOutputDirectory).grid(row=2, column=2)
setTemplateImageButton = tkinter.Button(swFrame, text = 'Set template image', bd = '5', command = chooseTemplate).grid(row=4, column=0)
startRegistrationButton = tkinter.Button(swFrame, text = 'Start single image registration', bd = '5', command = runSingleRegistration).grid(row=5, column=0)

quitButton = tkinter.Button(seFrame, text="Quit", bd = '5', command=root.destroy).grid(row=4, column=2)
# currentTemplateLabel = tkinter.Label(root, text=f'Template slice: {templateSliceNumber.get()}').grid(row=5,column=1)
root.mainloop()
