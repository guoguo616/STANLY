# STANLy
STANLy (**S**patial **T**ranscriptomic **A**lignment **N**on**L**inearl**y**) is a set of tools that utilizes the imaging data from spatial transcriptomic experiments such as Visium to register multiple images from into a common coordinate space to allow a truly spatial analysis of transcriptomic data. This toolbox is built in Python using tools from [Advanced Normalization Tools (ANTs)](http://stnava.github.io/ANTs/) and [The Allen Software Development Kit (SDK)](https://allensdk.readthedocs.io/en/latest/). Once aligned into a common space, we create 'digital' spots for each sample based on a selection of nearest neighbors and use these spots to run differential statistics on the experiment.

# Data Structure
Data can be arranged into a format based on [Brain Imaging Data Structure (BIDS)](bids.neuroimaging.io/), so that within an experimental folder there is a code folder for any code used to analyze the experiment, a derivatives folder which will contain all processed images and data, and a rawdata folder which contains the spatial folder and .h5 file from [SpaceRanger](https://support.10xgenomics.com/spatial-gene-expression/software/pipelines/latest/what-is-space-ranger) for each sample, i.e.:

    rawdata ->
        sample-01 ->
            spatial
            Sample1_SD_filtered_feature_bc_matrix.h5
        sample-02
        sample-03
        ...
        ...

# Registration
Images are prepared for analysis by first creating rotating the images into the same orientation as the template. Whenever a change is performed to the image we must also transform the corresponding spots, which we can do by applying the transformation acquired from the image registration to the XY coordinates of the spots. We use the Allen Common Coordinate Framework for our analysis, and in doing so aligned our images to the corresponding Nissl stained CCF image. Since the images collected by Visium are hematoxylin and eosin (H&E) stained we have used the inverted greyscale image in our registration, as this better matched the intensity distribution of the CCF Nissl Stains.

# Analysis
