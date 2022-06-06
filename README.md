# visiumAlignment

this projects is based around the 'extractSpatialInfoMultiSample.py' script in the 'code' folder

data should be arranged in 'pseudo-BIDS' format, so that there is rawdata folder which contains an experimental folder with the spaceranger data for each sample, i.e.

    rawdata ->
        sleepDep ->
            sample-01 ->
                spatial
                Sample1_SD_filtered_feature_bc_matrix.h5
                Sample1_SD_spatial.tar.gz
            sample-02
            sample-03
            ...
            ...

the list of genes to be searched can be changed at around lines 628 where there are already gene lists which can be input for the for loop ~line 640
