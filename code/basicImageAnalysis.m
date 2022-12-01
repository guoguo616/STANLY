%% import rawdata

close all; clear; clc
rawdata=fullfile('/','home','zjpeters','rdss_tnj','otherLabs','visiumAlignment','rawdata')

datafolder=fullfile('/','home','zjpeters','rdss_tnj','otherLabs','visiumAlignment','data')
imageSlice = imread(fullfile(datafolder,'allen10umSlices','allen10umcoronalLHSlice0700.png'));
tissue = imread(fullfile(rawdata,'sleepDepBothBatches','sample-01','spatial','tissue_hires_image.png'));
tissueRed = tissue(:,:,1);
tissueGreen = tissue(:,:,2);
tissueBlue = tissue(:,:,3);

%%
% histogram of green channel matches well with template
figure; histogram(tissueGreen)
figure; histogram(imageSlice)

maxGreen = double(max(max(tissueGreen)));
minGreen = double(min(min(tissueGreen)));

minMaxGreen = (double(tissueGreen) - minGreen)/(maxGreen - minGreen);
figure; histogram(minMaxGreen)

figure; histogram(1 - minMaxGreen)

greenHistMatched = 1 - minMaxGreen;
figure; imshow(greenHistMatched)