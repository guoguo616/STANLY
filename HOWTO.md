# Using STANLY to analyze data

The STANLY codebase is built in such a way that a user can either use it to register individual samples into Allen Common Coordinate Framework (CCF) and visualize the regional distribution of their spatial transcriptomic data, or to register an entire experiment worth of Visium samples and compare them to each other statistically.

# dependencies

List of dependencies can be found in requirements.txt. Additionally, this assumes you have run the Space Ranger pipeline on your data.

# Data Structure

STANLY was built with modularity involved, which is why we built the code around a mostly unified data formatting that allows processing to be performed on individual or multiple subjects easily. First, create an experimental folder that will contain your code, rawdata, derivatives, and sourcedata folders. Format your data in such a way that each sample in your experiment is given a unique identification of some sort, i.e. {sample-01,sample-02,...,sample-16}. If you have processed your Visium data locally and have the output of the `spaceranger count` separated into folders labeled with the sample ID, you can copy this into your rawdata folder. If you don't already have this prepared, you can create a folder for each sample and copy the `spatial` folder and `filtered_feature_bc_matrix.h5` file for each sample into its respective folder so that your experimental folder will look like this:

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

# Importing data

Create a python script in a code editor (I use Spyder through Anaconda) and save it to your code file. This will be the script that processes your data. With the STANLY code located somewhere you can easily access and import it and set the experimental folder for your analysis by running the `setExperimentalFolder` command.

    import stanly
    rawdata, derivatives = stanly.setExperimentalFolder("/home/user/data/sleepDeprivationVisium")
