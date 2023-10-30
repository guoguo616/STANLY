# How to use STANLY to analyze Visium spatial transcriptomic data

The STANLY codebase is built in such a way that a user can either use it to register individual spatial transcriptomic samples into Allen Common Coordinate Framework (CCF) and visualize the regional distribution of their spatial transcriptomic data, or to register an entire experiment worth of Visium samples and compare them to each other statistically. Note that these tools can be used to individually examine samples as well, but this how to will focus on how to prepare them for group analysis. This how to details the method used for statistical analysis presented in:
*Spatial transcriptomics reveals unique gene expression changes in different brain regions after sleep deprivation.*
Yann Vanrobaeys, Zeru Peterson, Emily Walsh, Snehajyoti Chatterjee, Li-Chun Lin, Lisa Lyons, Thomas Nickl-Jockschat, Ted Abel
bioRxiv 2023.01.18.524406; doi: https://doi.org/10.1101/2023.01.18.524406

This HOWTO focuses on analysis of Visium data, but the process is mainly the same for Merfish data
# Dependencies

List of dependencies can be found in requirements.txt. Additionally, this assumes you have run the Space Ranger pipeline on your Visium data.

~~You will need to make sure that ANTs is installed both on the system and for Python, hence why `ants` and `antspyx` are both listed.~~

Should now be updated so that you don't need to have `ants` installed, only `antspyx` (2023-10-30)
# Data Structure

STANLY was built with modularity in mind, which is why we built the code around a mostly unified data formatting that allows processing to be performed on individual or multiple subjects easily. First, create an experimental folder that will contain your code, rawdata, derivatives, and sourcedata folders. Format your data in such a way that each sample in your experiment is given a unique identification of some sort, i.e. {sample-01,sample-02,...,sample-16}. If you have processed your Visium data locally and have the output of the `spaceranger count` separated into folders labeled with the sample ID, you can copy this into your rawdata folder. If you don't already have this prepared, you can create a folder for each sample and copy the `spatial` folder and `filtered_feature_bc_matrix.h5` file for each sample into its respective folder so that your experimental folder will look like this (for the rest of the example, the location will be `/home/user/data/sleepDeprivationVisium`):

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

*Note: STANLY will do some brief searching to try and adjust for differences in folder structure, but this is under expansion*

# Participants file

Also inside of the rawdata folder you will need to create a .tsv file listing information about your samples including the sample ID, degrees of affine rotation, experimental group [0,1], and whether or not the sample should be flipped across the hemisphere [0,1].
```
participant_id  deg_rot sleep_dep flip
sample-01       270     1           0
sample-02       270     0           0
sample-03       270     1           0
... ... ...
sample-16       180    1           1
```
The degrees of affine rotation is the degrees of counter-clockwise rotation that it takes to bring a coronal slice into the same orientation as the template to be registered to. Additionally, to flip the hemisphere of a slice, set the value in the final column to 1, as with sample-16 above. These will all be inputs to the `processVisiumData` or `processMerfishData` functions. ***Text of image will be updated soon, the images showing -180 rotation are now 180 + flip***
![Image shows three examples of rotation performed in STANLY. First row, with the superior side of the mouse brain facing the right side of the screen a 90 degree counter-clockwise rotation is performed so that the superior side is facing the top of the screen. Second row, with the superior side of the mouse brain facing the left side of the screen a 270 degree counter-clockwise rotation is performed so that the superior side is facing the top of the screen. Third row, with the superior side of the mouse brain facing the bottom side of the screen a -180 degree counter-clockwise rotation is performed so that the superior side is facing the top of the screen and additionally the slice has been flipped across the left-right axis. All images are displayed in radiological orientation.](/source/images/rotationExplainer.png)*Examples for describing rotation in participants.tsv*
# Importing data

Create a python script in a code editor (I use Spyder through Anaconda) and save it inside your code folder. This will be the script you use to process your data. Assuming you have downloaded the STANLY code and stored it somewhere you can easily access and import it, the first command we will use from STANLY will be to set the experimental folder for your analysis by running the `setExperimentalFolder` command as shown below. (Alternately, you can set your `rawdata` and `derivatives` variables manually by using `os.path.join()` to assign a folder to each.)
```python
import os
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import sys
sys.path.insert(0, "/location/of/STANLY")
import stanly

rawdata, derivatives = stanly.setExperimentalFolder("/home/user/data/sleepDeprivationVisium")
# this is a list of the images to be included in the analysis based on visual inspection
# alternately, a user could not include the QC failed data in the folder and use something like:
# imageList = range(nOfSamples)
imageList = [0,1,2,3,4,5,6,7,10,11,12,13,15]
experiment = stanly.loadParticipantsTsv(os.path.join(rawdata, 'participants.tsv'), imageList)
```
If you have already removed any excluded samples from your `participants.tsv` file you can ignore the second option of the `loadParticipantsTsv` function and it will import all samples in `participants.tsv`.

# Loading the template and processing samples

STANLY has been built using the Allen Common Coordinate Framework (CCF) template image for registration of coronal brain slices from the mouse. To find the correct slice for alignment the user should look through the [Interactive Atlas Viewer](http://atlas.brain-map.org/atlas?atlas=1#atlas=1) and use the number of the slice listed below the image as the input for `chooseTemplateSlice` in STANLY. When run, this will first check if the user has already downloaded the reference data and if not download it. This may take awhile, so be patient during the loading. Once loaded, it will have the image data for the appropriate Nissl stained image as well as the annotation information provided by the Allen Institute. The next step is to process all of the samples imported above and output the processed data into our `derivatives` folder. We can do this all using a loop like so:
*Note: if you want to flip one of your samples, you can add `flip=True` option to the `processVisiumData` function*
```python
 template = stanly.chooseTemplateSlice(70)

 processedSamples = {}
totalSpotCount = 0
for actSample in range(len(experiment['sample-id'])):
    sample = stanly.importVisiumData(os.path.join(rawdata, experiment['sample-id'][actSample]))
    sampleProcessed = stanly.processVisiumData(sample, template, experiment['rotation'][actSample], derivatives)
    processedSamples[actSample] = sampleProcessed
    totalSpotCount += sampleProcessed['spotCount']
nTotalSamples = len(processedSamples)
spotCountMean = totalSpotCount / nTotalSamples
print(f"Average spot count across {nTotalSamples} samples is {spotCountMean}")
 ```

This loop additionally calculates the average number of spots from all samples for a sanity check. Will also show the image of spot-wise z-scores for total gene count.

# Register best fit image to template and apply transformation to remaining samples

From your experiment, select one sample with good image quality to be your "best fit" image for registration. This image will be registered to the Allen CCF template image using the `runANTsToAllenRegistration` function, and all other images will subsequently be registered to this best fit using the `runANTsInterSampleRegistration` function, and have the best fit-to-Allen transformation applied to their images and spots using the `applyAntsTransformations` function. For this data, the sample we selected as best fit is sample-05 (which has a Python index of [4] in our `processedSamples` variable created above). If your data contains only data from a particular hemisphere, you will want to use one of the options `hemisphere='rightHem'`, `hemisphere='leftHem'`, or `hemisphere='wholeBrain'` (now default) to the `runANTsToAllenRegistration` and `applyAntsTransformations`.

```python
bestSampleToTemplate = stanly.runANTsToAllenRegistration(processedSamples[4], template, hemisphere='rightHem')

experimentalResults = {}
for actSample in range(len(processedSamples)):
    sampleRegistered = stanly.runANTsInterSampleRegistration(processedSamples[actSample], bestSample)
    experimentalResults[actSample] = sampleRegistered

allSamplesToAllen = {}
for actSample in range(len(experimentalResults)):
    regSampleToTemplate = stanly.applyAntsTransformations(experimentalResults[actSample], bestSampleToTemplate, template)
    allSamplesToAllen[actSample] = regSampleToTemplate
```

# Check the results of registration

It is at this point that the user will want to carefully examine the quality of the registration, both to the template and between samples. Potential places for error include:
- flipped hemispheres (i.e. your slice does not match the hemispheric orientation of your other samples)
- check for the alignment of landmark regions in your images

# Generate digital spots for data

Using the aligned data, we will begin generating our digital spots. First we will create the 'template' digital spots, which tells us how many spots of a certain dimension will fit in our best fit sample. In the snippet below, we will generate spots at a diameter of 150um. Since STANLY expects spot size in 10um increments, we will set our input size to 15. With the template spot coordinates created we can then calculate the K-nearest neighbors from each sample for each digital spot, in this case `K=7`. We will also create a list of genes present in each sample within this same loop.
 ```python
 #%% create digital spots for whole slice and find nearest neighbors
kSpots = 7
templateDigitalSpots = stanly.createDigitalSpots(bestSampleToTemplate, 15)
allSampleGeneList = allSamplesToAllen[0]['geneListMasked']
for i, regSample in enumerate(allSamplesToAllen):        
    actNN, actCDist = stanly.findDigitalNearestNeighbors(templateDigitalSpots, allSamplesToAllen[i]['maskedTissuePositionList'], kSpots, wholeBrainSpotSize)
    allSamplesToAllen[i]['digitalSpotNearestNeighbors'] = np.asarray(actNN, dtype=int)
    # creates a list of genes present in all samples
    if i == 0:
        continue
    else:
        allSampleGeneList = set(allSampleGeneList) & set(allSamplesToAllen[i]['geneListMasked'])
# some spot and group numbers that will be used later
nDigitalSpots = len(templateDigitalSpots)
nSampleExperimental = sum(experiment['experimental-group'])
nSampleControl = len(experiment['experimental-group']) - nSampleExperimental
```
# Running statistics on digital spots
Now that we have our digital spots, we will run a t-statistic on the log-2 normalized data in each digital spot. The code below does a decent amount, including running the t-test, checking for significance, and outputting any significant genes along with t-statistic and p-value information for that gene at each spot coordinate in the template digital spots. It additionally outputs an image containing the mean value at each spot between groups, along with the t-statistic for any significant genes. The significant p-value is calculated using Sidak correction, $$\alpha_s  = 1 - (1 - \alpha)^{\frac{1}{m*n}}$$ where $\alpha$ is the starting p-value, $m$ is the number of digital spots, and $n$ is the number of genes in the list.

```python
#%% run t-test using Sidak corrected p-value
start_time = time.time()
desiredPval = 0.05
alphaSidak = 1 - np.power((1 - desiredPval),(1/(nDigitalSpots*len(allSampleGeneList))))

sigGenes = []
sigGenesWithPvals = []
sigGenesWithTstats = []
# loops over each gene to search for significantly different spots
for nOfGenesChecked,actGene in enumerate(allSampleGeneList):
    digitalSamplesControl = np.zeros([nDigitalSpots,(nSampleControl * kSpots)])
    digitalSamplesExperimental = np.zeros([nDigitalSpots,(nSampleExperimental * kSpots)])
    startControl = 0
    stopControl = kSpots
    startExperimental = 0
    stopExperimental = kSpots
    nTestedSamples = 0
    nControls = 0
    nExperimentals = 0
    # this splits each sample into its experimental group 
    for actSample in range(len(allSamplesToAllen)):
        ## if you are going to be using a self defined gene list, you should uncomment and use the below section instead of the included geneIndex line
        # try:
        #     geneIndex = allSamplesToAllen[actSample]['geneListMasked'].index(actGene)
        # except(ValueError):
        #     print(f'{actGene} not in dataset')
        #     continue
        geneIndex = allSamplesToAllen[actSample]['geneListMasked'].index(actGene)
        geneCount = np.zeros([nDigitalSpots,kSpots])
        for spots in enumerate(allSamplesToAllen[actSample]['digitalSpotNearestNeighbors']):
            if np.any(spots[1] < 0):
                geneCount[spots[0]] = np.nan
            else:
                spotij = np.zeros([7,2], dtype=int)
                spotij[:,0] = geneIndex
                spotij[:,1] = np.asarray(spots[1], dtype=int)
                geneCount[spots[0]] = allSamplesToAllen[actSample]['filteredFeatureMatrixMasked'][spotij[:,0],spotij[:,1]]
                
        spotCount = np.nanmean(geneCount, axis=1)
        nTestedSamples += 1
        if experiment['experimental-group'][actSample] == 0:
            digitalSamplesControl[:,startControl:stopControl] = geneCount
            startControl += kSpots
            stopControl += kSpots
            nControls += 1
        elif experiment['experimental-group'][actSample] == 1:
            digitalSamplesExperimental[:,startExperimental:stopExperimental] = geneCount
            startExperimental += kSpots
            stopExperimental += kSpots
            nExperimentals += 1
        else:
            continue
        
    digitalSamplesControl = np.array(digitalSamplesControl, dtype='float32').squeeze()
    digitalSamplesExperimental = np.array(digitalSamplesExperimental, dtype='float32').squeeze()
    # this will check that at least a certain number of spots show expression for the gene of interest
    checkControlSamples = np.count_nonzero(digitalSamplesControl,axis=1)
    checkExperimentalSamples = np.count_nonzero(digitalSamplesExperimental,axis=1)
    checkAllSamples = checkControlSamples & checkExperimentalSamples > 0
    if sum(checkAllSamples) < 20:
        continue
    else:
        maskedDigitalSamplesControl = np.zeros(digitalSamplesControl.shape)
        maskedDigitalSamplesExperimental = np.zeros(digitalSamplesExperimental.shape)
        maskedDigitalSamplesControl[checkAllSamples,:] = digitalSamplesControl[checkAllSamples,:]
        maskedDigitalSamplesExperimental[checkAllSamples,:] = digitalSamplesExperimental[checkAllSamples,:]
        maskedTtests = []
        allTstats = np.zeros(nDigitalSpots)
        actTtest = scipy.stats.ttest_ind(digitalSamplesExperimental,digitalSamplesControl, axis=1, nan_policy='propagate')
        actTstats = actTtest[0]
        actPvals = actTtest[1]
        mulCompResults = actPvals < alphaSidak
        if sum(mulCompResults) > 0:
            actSigGene = [actGene,sum(mulCompResults)]
            sigGenes.append(actSigGene)
            actSigGeneWithPvals = np.append(actSigGene, actPvals)
            actSigGeneWithTstats = np.append(actSigGene, actTstats)
            sigGenesWithPvals.append(actSigGeneWithPvals)
            sigGenesWithTstats.append(actSigGeneWithTstats)
            maskedDigitalCoordinates = templateDigitalSpots[np.array(mulCompResults)]
            maskedTstats = actTtest[0][mulCompResults]
            maskedDigitalCoordinates = np.array(maskedDigitalCoordinates)
            meanDigitalControl = np.nanmean(digitalSamplesControl,axis=1)
            meanDigitalExperimental = np.nanmean(digitalSamplesExperimental,axis=1)
            finiteMin = np.nanmin(actTtest[0])
            finiteMax = np.nanmax(actTtest[0])
            maxGeneCount = np.nanmax([medianDigitalControl,medianDigitalExperimental])
            #Plot data
            fig, axs = plt.subplots(1,3)
            plt.axis('off')
            axs[0].imshow(allSamplesToAllen[4]['tissueRegistered'],cmap='gray',aspect="equal")
            axs[0].scatter(templateDigitalSpots[:,0],templateDigitalSpots[:,1], c=np.array(meanDigitalControl), vmin=0,vmax=3,plotnonfinite=False,cmap='Reds',marker='.')
            axs[0].set_title('Control')
            axs[0].axis('off')
            axs[1].imshow(allSamplesToAllen[4]['tissueRegistered'],cmap='gray',aspect="equal")
            axs[1].scatter(templateDigitalSpots[:,0],templateDigitalSpots[:,1], c=np.array(meanDigitalExperimental), vmin=0,vmax=3,plotnonfinite=False,cmap='Reds',marker='.')
            axs[1].set_title('Experimental')
            axs[1].axis('off')
            axs[2].scatter(templateDigitalSpots[:,0],templateDigitalSpots[:,1], c=np.array(actTstats), cmap='seismic', vmin=-4,vmax=4,plotnonfinite=False,marker='.')
            axs[2].imshow(allSamplesToAllen[4]['tissueRegistered'],cmap='gray')
            axs[2].set_title(actGene, style='italic')
            axs[2].axis('off')

            plt.savefig(os.path.join(derivatives,f'tStatGeneCount{actGene}SleepDep.png'), bbox_inches='tight', dpi=300)
            plt.show()

timestr = time.strftime("%Y%m%d-%H%M%S")
with open(os.path.join(derivatives,f'listOfSigGenesSidakPvalues_{nameForMask}_{timestr}.csv'), 'w', encoding='UTF8') as f:
    writer = csv.writer(f)
    for i in sigGenesWithPvals:
        writer.writerow(i)
        
with open(os.path.join(derivatives,f'listOfSigGenesSidakTstatistics_{nameForMask}_{timestr}.csv'), 'w', encoding='UTF8') as f:
    writer = csv.writer(f)
    for i in sigGenesWithTstats:
        writer.writerow(i)
        
print("--- %s seconds ---" % (time.time() - start_time))

```

Following this, you should have a number of images for the genes of interest, along with two csv files containing the p-value and t-statistic information for your data. ![Image shows results from the above analysis for the gene Camk2n1 between non-sleep-deprived and sleep deprived conditions, showing the mean results for the mean of the non-sleep-deprived digital spots on the left in a red gradient, the mean results of the sleep-deprived digital spots in the center in a red gradient, and the t-statistics for each spot on the right in a blue to red gradient, where blue indicates a negative t-statistic and red indicates a positive t-statistic.](/source/images/tStatGeneCountCamk2n1SleepDep.png)*Example results from above analysis for gene Camk2n1 between control and sleep deprived groups.*

## Descriptions of expected outputs
- `*_tissuePointsProcessed.csv` - a list of the [x,y] coordinates of each tissue containing spot
- `*_tissue.png` - image of tissue slice masked before pre-processing
- `*_tissueProcessed.png` - pre-processed greyscale image of tissue slice to be used in registration process
- `*_tissuePointOrderedFeatureMatrixLog2Normalized.npz` - numpy compressed array containing gene matrix information
- `*_processing_information.json` - json file containing information used in the processing and registration process, as well as gene list for sample