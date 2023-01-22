# How to use STANLY to analyze Visium spatial transcriptomic data

The STANLY codebase is built in such a way that a user can either use it to register individual samples into Allen Common Coordinate Framework (CCF) and visualize the regional distribution of their spatial transcriptomic data, or to register an entire experiment worth of Visium samples and compare them to each other statistically.

# Dependencies

List of dependencies can be found in requirements.txt. Additionally, this assumes you have run the Space Ranger pipeline on your data.

# Data Structure

STANLY was built with modularity involved, which is why we built the code around a mostly unified data formatting that allows processing to be performed on individual or multiple subjects easily. First, create an experimental folder that will contain your code, rawdata, derivatives, and sourcedata folders. Format your data in such a way that each sample in your experiment is given a unique identification of some sort, i.e. {sample-01,sample-02,...,sample-16}. If you have processed your Visium data locally and have the output of the `spaceranger count` separated into folders labeled with the sample ID, you can copy this into your rawdata folder. If you don't already have this prepared, you can create a folder for each sample and copy the `spatial` folder and `filtered_feature_bc_matrix.h5` file for each sample into its respective folder so that your experimental folder will look like this (for the rest of the example, the location will be `/home/user/data/sleepDeprivationVisium`):

    sleepDeprivationVisium ->
      rawdata ->
          sample-01 ->
              Sample1_SD_filtered_feature_bc_matrix.h5
              spatial ->
                  aligned_fiducials.jpg
                  detected_tissue_image.jpg
                  scalefactors_json.json
                  tissue_hires_image.png
                  tissue_lowres_image.png
                  tissue_positions_list.csv
          sample-02 ->
              Sample2_SD_filtered_feature_bc_matrix.h5
              spatial
          sample-03
          ...

# Participants file

Also inside of the rawdata folder you will need to create a .tsv file listing information about your samples including the sample ID, degrees of affine rotation [0,+/-90,+/-180,+/-270], and experimental group [0,1].
```
participant_id  deg_rot sleep_dep
sample-01 270  1
sample-02 270 0
sample-03 270  1
... ... ...
sample-16 -180  1
```
The degrees of affine rotation is the degrees of counter-clockwise rotation that it takes to bring a coronal slice into the same orientation as the template to be registered to. Additionally, to flip the hemisphere of a slice, add a `-` before the degrees of rotation, as with sample-16 above.
![Image shows three examples of rotation performed in STANLY. First row, with the superior side of the mouse brain facing the right side of the screen a 90 degree counter-clockwise rotation is performed so that the superior side is facing the top of the screen. Second row, with the superior side of the mouse brain facing the left side of the screen a 270 degree counter-clockwise rotation is performed so that the superior side is facing the top of the screen. Third row, with the superior side of the mouse brain facing the bottom side of the screen a -180 degree counter-clockwise rotation is performed so that the superior side is facing the top of the screen and additionally the slice has been flipped across the left-right axis. All images are displayed in radiological orientation.](/source/images/rotationExplainer.png)*Examples for describing rotation in participants.tsv*
# Importing data

Create a python script in a code editor (I use Spyder through Anaconda) and save it to your code folder. This will be the script you use to process your data. Assuming you have downloaded the STANLY code and stored it somewhere you can easily access and import it. The first command we will use from STANLY will be to set the experimental folder for your analysis by running the `setExperimentalFolder` command.
```python
import os
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import stanly

rawdata, derivatives = stanly.setExperimentalFolder("/home/user/data/sleepDeprivationVisium")
# this is a list of the images to be included in the analysis based on visual inspection
imageList = [0,1,2,3,4,5,6,7,10,11,12,13,15]
experiment = stanly.loadParticipantsTsv(os.path.join(rawdata, 'participants.tsv'), imageList)
```
If you have already removed any excluded samples from your `participants.tsv` file you can ignore the second option of the `loadParticipantsTsv` function and it will import all samples in `participants.tsv`.
