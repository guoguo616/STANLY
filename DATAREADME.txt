This dataset contains spatially resolved probe based transcriptomic data from 16 male adult mice acquired using Visium from 10x Genomics. Single hemisphere samples were collected coronally from approximately -1.70mm Bregma according to Paxinos figure 45 by members of Ted Abel's (etc?) Laboratory at the University of Iowa.

* The dataset contains data from 16 experiments, 1 sample each per animal, containing an image of the tissue collected (10um thick sections of one coronal brain hemisphere) 
    GSM6922983 - Sample1SD - corresponds to a male mouse which underwent sleep deprivation
    GSM6922984 - Sample2NSD - corresponds to a male mouse under normal sleep conditions
    GSM6922985 - Sample3SD - corresponds to a male mouse which underwent sleep deprivation
    GSM6922986 - Sample4NSD - corresponds to a male mouse under normal sleep conditions
    GSM6922987 - Sample5SD - corresponds to a male mouse which underwent sleep deprivation
    GSM6922988 - Sample6NSD - corresponds to a male mouse under normal sleep conditions
    GSM6922989 - Sample7SD - corresponds to a male mouse which underwent sleep deprivation
    GSM6922990 - Sample8NSD - corresponds to a male mouse under normal sleep conditions
    GSM6922991 - Sample9NSD - corresponds to a male mouse under normal sleep conditions
    GSM6922992 - Sample10SD - corresponds to a male mouse which underwent sleep deprivation
    GSM6922993 - Sample11NSD - corresponds to a male mouse under normal sleep conditions
    GSM6922994 - Sample12SD - corresponds to a male mouse which underwent sleep deprivation
    GSM6922995 - Sample13NSD - corresponds to a male mouse under normal sleep conditions
    GSM6922996 - Sample14SD - corresponds to a male mouse which underwent sleep deprivation
    GSM6922997 - Sample15NSD - corresponds to a male mouse under normal sleep conditions
    GSM6922998 - Sample16SD - corresponds to a male mouse which underwent sleep deprivation

* A total of up to 32285 genes were mapped onto up to 4,992 probes per Visium slide

* The rawdata folder contains 16 folders, 1 folder per sample, each named with the associated subject ID. These folders contain, for each subject, the filtered_feature_bc_matrix.h5 file containing the filtered feature barcode matrix, as well as a folder titled spatial containing output from the SpaceRanger processing pipeline. The spatial folder for each subject contains the following images: aligned_fiducials.jpg, detected_tissue_image.jpg, tissue_hires_image.png, tissue_lowres_image.png, as well as a csv file named tissue_positions_list.csv and a json file named scalefactors_json.json

* The derivatives folder contains 13 folders, 1 folder per sample processed with the STANLY pipeline, each named with the associated subject ID
