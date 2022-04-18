#!/bin/bash

#this directory format only works if run from the code directory
rawdata=../rawdata
derivatives=../derivatives
#experimental=
# template=/home/zjpeters/Documents/mouseDevelopmental/templates/waxholm_subsamp8_Anat_Iso.nii


while read participant_id age sex genotype; do
  [ "$participant_id" == participant_id ] && continue;  # skips the header
  antsRegistrationSyN.sh -d 2 -f allenSlice73.png -m sorSlice1.png -o sor2allen
done < $rawdata/participants.tsv

## testing transformation
antsApplyTransformsToPoints -d 2 -i inTissueDotsRotated.csv -o test.csv -t [ sor2allen0GenericAffine.mat,1]
