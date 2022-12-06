%% import rawdata

close all; clear; clc
rawdata=fullfile('/','home','zjpeters','Documents','visiumalignment','rawdata');

datafolder=fullfile('/','home','zjpeters','Documents','visiumalignment','data');
imageSlice = imread(fullfile(datafolder,'allen10umSlices','allen10umLHSlice0700.png'));
tissue = imread(fullfile(rawdata,'sample-01','spatial','tissue_hires_image.png'));
tissueRed = tissue(:,:,1);
tissueGreen = tissue(:,:,2);
tissueBlue = tissue(:,:,3);

%%
% histogram of green channel matches well with template
figure; histogram(tissueGreen); title('Histogram of Green Channel');
figure; histogram(imageSlice); title('Histogram of Allen Slice');

maxGreen = double(max(max(tissueGreen)));
minGreen = double(min(min(tissueGreen)));

minMaxGreen = (double(tissueGreen) - minGreen)/(maxGreen - minGreen);
figure; histogram(minMaxGreen)

figure; histogram(1 - minMaxGreen)

greenHistMatched = 1 - minMaxGreen;
figure; imshow(greenHistMatched); title('Histogram of [1 - Green Channel]');
%%

tissueGrey = rgb2gray(tissue);
figure; histogram(tissueGrey); title('Histogram of greyscale Visium');
figure; histogram(imageSlice); title('Histogram of greyscale Allen');

maxGrey = double(max(max(tissueGrey)));
minGrey = double(min(min(tissueGrey)));

minMaxGrey = (double(tissueGrey) - minGrey)/(maxGrey - minGrey);
figure; histogram(minMaxGrey)

figure; histogram(1 - minMaxGrey); title('Histogram of [1 - Visium]');

greyHistMatched = 1 - minMaxGrey;
figure; imshow(greyHistMatched)
