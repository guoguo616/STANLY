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
# open window with set dimensions
root = Tk()
root.title('STANLy')
root.geometry('200x200')
# filedialog.askdirectory()

def loadExperiment():
    experimentPath = filedialog.askdirectory()
    return experimentPath


# Create a Button
loadExperimentButton = tkinter.Button(root, text = 'Load experiment', bd = '5', command = loadExperiment).pack()
# loadExperimentButton.pack(side = 'top')
loadSampleButton = tkinter.Button(root, text = 'Load sample', bd = '5', command = filedialog.askdirectory).pack()
setDerivativesButton = tkinter.Button(root, text = 'Set output directory', bd = '5', command = filedialog.askdirectory).pack()
# loadSampleButton.pack(side = 'bottom')
root.mainloop()
