%% extract information from spaceranger output
clear; close all; clc

derivatives = fullfile('/','media','zjpeters','Samsung_T5','visiumAlignment','derivatives');
addpath(fullfile('/','home','zjpeters','matlabToolboxes','JSONio-main'));
addpath(fullfile('/','home','zjpeters','matlabToolboxes','loadSVG'));
templateImage = imread(fullfile('/','media','zjpeters','Samsung_T5','visiumAlignment','data','spatial','mouse_coronal_nissl_slice_73.jpg'));
% full tissue position list information [barcode,inTissue,arrayXPosix,arrayYPosix,xPosix,yPosix 
tissuePositionsList = readtable(fullfile('/','media','zjpeters','Samsung_T5','visiumAlignment','data','spatial','Sample1SOR_spatial','spatial','tissue_positions_list.csv'));
tissuePositionsList.Properties.VariableNames = {'barcode','inTissue','arrayXPosix','arrayYPosix','xPosix','yPosix'};
% contains [tissue_hires_scalef,tissue_lowres_scalef,fiducial_diameter_fullres,spot_diameter_fullres]
sampleJson = jsonread(fullfile('/','media','zjpeters','Samsung_T5','visiumAlignment','data','spatial','Sample1SOR_spatial','spatial','scalefactors_json.json'));
highRes = imread(fullfile('/','media','zjpeters','Samsung_T5','visiumAlignment','data','spatial','Sample1SOR_spatial','spatial','tissue_hires_image.png'));
highResX = size(highRes,1);
highResY = size(highRes,2);
realX = ceil(highResX / sampleJson.tissue_hires_scalef);
realY = ceil(highResY / sampleJson.tissue_hires_scalef);
h5disp(fullfile('/','media','zjpeters','Samsung_T5','visiumAlignment','data','spatial','Sample1SOR_filtered_feature_bc_matrix.h5'));

% atlas image is originally over the right hemisphere, needs to be flipped
atlasImageRightHemisphere = imread(fullfile('/','media','zjpeters','Samsung_T5','visiumAlignment','data','mouse_coronal_slice73.png'));
atlasImageFlip = flipdim(atlasImageRightHemisphere, 2);
%% resize template and visium image into same space

templateImageHeight = size(templateImage,1);

templateImageWidth = size(templateImage,2);
% cut the image so it's only the part that matches the visium slice
templateImageLeftHemisphere = templateImage(:,1:(templateImageWidth/2 - 200),:);
atlasImageLeftHemisphere = atlasImageFlip(:,1:(templateImageWidth/2 - 200),:);

atlasImageResize = imresize(atlasImageLeftHemisphere,[1800,NaN]);
atlasImageCentered = padarray(atlasImageResize, [100,393],'both');

templateImageResize = imresize(templateImageLeftHemisphere,[1800,NaN]);
templateImageGrey = rgb2gray(templateImageResize);
templateImageBin = templateImageGrey;
% templateImageNoBackground(templateImageGrey > 200) = 0;
templateImageGaussian = imgaussfilt(templateImageGrey, 16);
templateImageNoBackground = templateImageGaussian; templateImageNoBackground(templateImageGaussian > 230) = 0;
templateImageBin = imbinarize(templateImageNoBackground); templateImageBin = imfill(templateImageBin,'holes');   % fills dots of border
templateImageBorder = bwareafilt(templateImageBin, [0,1000]); % remove the tissue slice to search for fiducials
templateImageSliceMask = bwareafilt(templateImageBin, 1);
dilateSe = strel('disk',50);
templateImageSliceMaskDilate = imdilate(templateImageSliceMask,dilateSe);
erodeSe = strel('disk',40);
templateImageSliceMaskErode = imerode(templateImageSliceMaskDilate, erodeSe);
% display color channels of template image
templateImageSlice = double(templateImageGrey) .* double(templateImageSliceMaskErode);
templateImageSliceCentered = padarray(templateImageSlice, [100,393],'both');
figure; imshow(templateImageSliceCentered)
figure; subplot(2,2,1); imshow(templateImageResize); subplot(2,2,2); imshow(templateImageResize(:,:,1)); subplot(2,2,3); imshow(templateImageResize(:,:,2)); subplot(2,2,4); imshow(templateImageResize(:,:,3));
figure; imshowpair(templateImageGaussian,templateImageSliceMaskDilate)
templateFilename = fullfile(derivatives,'allen_slice_73_tissue');
niftiwrite(uint8(templateImageSliceCentered), templateFilename);


highResGrey = rgb2gray(highRes);
highResBin = ~imbinarize(highResGrey); highResBin = imfill(highResBin,'holes');   % fills dots of border
highResBorder = bwareafilt(highResBin, [0,1000]); % remove the tissue slice to search for fiducials
highResScliceMask = bwareafilt(highResBin, 1);
highResSlice = double(highResGrey) .* double(highResScliceMask);
% highResSliceRotate = imrotate(highResSlice, 180);
figure; imshowpair(templateImageSliceCentered, highResSlice);
highResFilename = fullfile(derivatives,'sor_slice_1_tissue');
niftiwrite(uint8(highResSlice), highResFilename);
%% plot in tissue points as defined by spaceranger output
tissueMatrix = zeros(realX,realY);
inTissueCoor = [];
n = 0;
for i = 1:height(tissuePositionsList)
    if tissuePositionsList.inTissue(i) == 1
        n = n + 1;
        inTissueCoor(n, :) = [tissuePositionsList.xPosix(i),tissuePositionsList.yPosix(i)];
    else
        continue
    end
end

% plots the in tissue coordinates as output by spaceranger
figure; imshowpair(templateImageGrey,highResGrey,'montage')
figure; plot(inTissueCoor(:,2),inTissueCoor(:,1), 'o'); set(gca,'YDir','reverse')
% figure; imshow(templateImageResize)
