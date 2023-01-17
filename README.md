# STANLy
STANLy (**S**patial **T**ranscriptomic **A**lignment **N**on**L**inearl**y**) is a set of tools that utilizes the imaging data from spatial transcriptomic experiments such as Visium to register multiple images from into a common coordinate space to allow a truly spatial analysis of transcriptomic data. This toolbox is built in Python using tools from [Advanced Normalization Tools (ANTs)](http://stnava.github.io/ANTs/) and [The Allen Software Development Kit (SDK)](https://allensdk.readthedocs.io/en/latest/). Once aligned into a common space, we create 'digital' spots for each sample based on a selection of nearest neighbors and use these spots to run differential statistics on the experiment.
![Image shows a visual representation of registering a Visium sample slice, including tissue image and spot data, into a template space. Image also shows 6 example differentially expressed genes.](/source/images/figure5portraitWhiteBG230106.png)*Figure5*


# Data Structure
Data can be arranged into a format based on [Brain Imaging Data Structure (BIDS)](bids.neuroimaging.io/), so that within an experimental folder there is a code folder for any code used to analyze the experiment, a derivatives folder which will contain all processed images and data, and a rawdata folder which contains VIsium data, i.e. the spatial folder and .h5 file from [SpaceRanger](https://support.10xgenomics.com/spatial-gene-expression/software/pipelines/latest/what-is-space-ranger) for each sample, i.e.:

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
        sample-02
        sample-03
        ...
        ...

# Registration
Images are prepared for analysis by first creating rotating the images into the same orientation as the template. Whenever a change is performed to the image we must also transform the corresponding spots, which we can do by applying the transformation acquired from the image registration to the XY coordinates of the spots. We use the Allen Common Coordinate Framework for our analysis, and in doing so aligned our images to the corresponding Nissl stained CCF image. Since the images collected by Visium are hematoxylin and eosin (H&E) stained we have used the inverted greyscale image in our registration, as this better matched the intensity distribution of the CCF Nissl Stains.

# Code description
- stanly.py: contains the codebase for STANLY. This includes functions for importing visium data, importing template data from the Allen CCF nrrd files, rotating Visium tissue points into proper position, processing Visium image and transcriptomic data, registering individual samples to other samples, registering individual samples to CCF, applying the CCF transformations to samples, creating digital spots, calculating the nearest neighbors for digital spots, creating regional masks for analysis based on Allen regional parcellation, loading gene lists from txt or csv, loading samples that have already been processed and registered in STANLY, and running basic statistics on the digital spots.
- stanlyGUI.py: **IN DEVELOPMENT** a gui built upon the stanly.py codebase that allows users to view, process, register, and interact with the data in a windowed environment
- stanlyOnSleepDep.py: script used to run full STANLy processing and analysis on the sleep deprivation dataset
- extractStanlyInfoFromRegisteredSamples.py: script that runs the analysis on the already processed and registered sleep dep samples
- runGeneListThroughToppGene.sh: bash script that takes the gene list output by the STANLy analysis and runs them through ToppGene functional enrichment analysis

# License
STANLY is released under the GPL v3 license. Commercial use of this software is prohibited due to licensing of software components used.
