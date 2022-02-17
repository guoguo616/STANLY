%% a script to align visium slices to some template space
% mostly taken from 'binTif210621.m'
close all; clear; clc
%prepare functions, templates, and fiducials and set defaults
fiducialDir = fullfile('/','media','zjpeters','Samsung_T5','visiumAlignment','data','fiducials');
scale=0.1;  % amount the images will be scaled for alignment
pixDim=(6.5/9000) * scale;  %mm/pixel resolution of template image, 9000 pixels between border, scaled to data
sliceThickness = 0.001 * scale; %thickness of each slice. 0.001 = 1 micron
templateImage = imresize(imread(fullfile('/','media','zjpeters','Samsung_T5','visiumAlignment','data','tinyimage.jpg')), scale);

% load fiducial images
upperLeftFiducial = imresize(imread(fullfile(fiducialDir,'upperLeftFiducial.png')),scale);
lowerLeftFiducial = imresize(imread(fullfile(fiducialDir,'lowerLeftFiducial.png')),scale);
upperRightFiducial = imresize(imread(fullfile(fiducialDir,'upperRightFiducial.png')),scale);
lowerRightFiducial = imresize(imread(fullfile(fiducialDir,'lowerRightFiducial.png')),scale);

upperLeftFiducialGrey = rgb2gray(upperLeftFiducial); upperLeftFiducialBin = ~imbinarize(upperLeftFiducialGrey); upperLeftFiducialBin = double(imfill(upperLeftFiducialBin,'holes'));
upperRightFiducialGrey = rgb2gray(upperRightFiducial); upperRightFiducialBin = ~imbinarize(upperRightFiducialGrey); upperRightFiducialBin = double(imfill(upperRightFiducialBin,'holes'));
lowerLeftFiducialGrey = rgb2gray(lowerLeftFiducial); lowerLeftFiducialBin = ~imbinarize(lowerLeftFiducialGrey); lowerLeftFiducialBin = double(imfill(lowerLeftFiducialBin,'holes'));
lowerRightFiducialGrey = rgb2gray(lowerRightFiducial); lowerRightFiducialBin = ~imbinarize(lowerRightFiducialGrey); lowerRightFiducialBin = double(imfill(lowerRightFiducialBin,'holes'));

%% import data to be aligned, resize
% resize templateImage and convert to grayscale
imageDir = fullfile('./data/Visium_Round 2_Abel_Lab_Nov_2021/'); % location of tifs to be aligned
allImages = struct2table(dir(fullfile(imageDir,'*.tif')));
numImages = height(allImages);

for i=1:numImages
    filename=cell2mat(fullfile(allImages.folder(i),allImages.name(i)));
    I = imread(filename);
    v = size(templateImage, 1)/size(I,1);   % calculates an image specific constant to resize to the template
    ir = imresize(I,v);
    im{i}=ir;
    clear filename I v ir;
end
gIm=cellfun(@rgb2gray,im,'UniformOutput',false);

%% prepare images for alignment

for i = 1:length(gIm)
    tem = ~imbinarize(gIm{i}); tem = imfill(tem,'holes');   % fills dots of border
    imBord = bwareafilt(tem, [0,1000]); % remove the tissue slice to search for fiducials
    slice = bwareafilt(tem, 1);
    bord{i} = double(imBord);   % extracted borders of each image
    sliceMask{i} = slice;
    clear tem imBord slice
end

%% find fiducials
% finds proper orientation of images in relation to the template
aR = [0, 90, 180, 270];
for i = 1:length(bord)
    n = 1;
    for r = 1:4
        act = imrotate(bord{i}, aR(r));
        cul = normxcorr2(upperLeftFiducialBin, act);% searches for fiducials in each orientation
        C(1, n) = max(cul(:));  % extracts max match for each fiducial
        cur = normxcorr2(upperRightFiducialBin, act);
        C(2, n) = max(cur(:));
        cll = normxcorr2(lowerLeftFiducialBin, act);
        C(3, n) = max(cll(:));
        clr = normxcorr2(lowerRightFiducialBin, act);
        C(4, n) = max(clr(:));

        n = n + 1;
    end
    C = mean(C);    % finds the mean matching of all fiducials per orientation
    [mc, midx] = max(C);    % finds the best fit rotation
    rotBord{i} = imrotate(bord{i}, aR(midx));   % performs proper rotation of border image
    rotIm{i} = imrotate(gIm{i}, aR(midx));  % performs proper rotation of grayscale image
    rotSliceMask{i} = imrotate(double(sliceMask{i}), aR(midx));  % performs proper rotation of slice image
    clear act C cul cur cll clr
end

% min-max normalization

for i = 1:length(rotIm)
    act = [];
    act = im2double(rotIm{i});
    act = (act - min(act(:))) / (max(act(:)) - min(act(:)));
    act = act .* rotSliceMask{i};
    rotImNorm{i} = act;
end
figure; montage(rotImNorm)

%% match image sizes without changing pixel info

[s,i]=cellfun(@size,rotImNorm);
maxh=max(s);                    %find maximum pixel height of images
maxw=max(i);                    %fine maximum pixel width of images
rotImNormPad = rotImNorm;
rotSliceMaskPad = rotSliceMask;
for i=1:length(rotImNormPad)
    h=size(rotImNormPad{i},1);
    w=size(rotImNormPad{i},2);
    if h < maxh
      hd=maxh-h;
      hpa=floor(hd/2);
      rotImNormPad{i}=padarray(rotImNormPad{i},[hpa,0],0,'both');
      rotSliceMaskPad{i} = padarray(rotSliceMaskPad{i}, [hpa,0],0,'both');
      if size(rotImNormPad{i},1)==(maxh-1)
          rotImNormPad{i}=padarray(rotImNormPad{i},[1,0],0,'pre');
          rotSliceMaskPad{i} = padarray(rotSliceMaskPad{i},[1,0],0,'pre');
      else
      end
    else
        rotImNormPad{i}=rotImNormPad{i};
        rotSliceMaskPad{i} = rotSliceMaskPad{i};
    end
    if w < maxw
        wd=maxw-w;
        wpa=floor(wd/2);
        rotImNormPad{i}=padarray(rotImNormPad{i},[0,wpa],0,'both');
        rotSliceMaskPad{i} = padarray(rotSliceMaskPad{i}, [0,wpa],0,'both');
        if size(rotImNormPad{i},2)==(maxw-1)
            rotImNormPad{i}=padarray(rotImNormPad{i},[0,1],0,'pre');
            rotSliceMaskPad{i} = padarray(rotSliceMaskPad{i}, [0,1],0,'pre');
        else
        end
    else
        rotImNormPad{i}=rotImNormPad{i};     %size corrected images contained in cell
        rotSliceMaskPad{i} = rotSliceMaskPad{i};
    end
end


%% find centroid of each image

for i = 1:length(rotSliceMaskPad)
    act = [];
    act = rotSliceMaskPad{i};  % use binary images to find centroid
    stat = regionprops(act,'Centroid');
    centroid(i,1) = stat.Centroid(1); centroid(i,2) = stat.Centroid(2); % create a variable containing all centroids
end

%% align images to have similar centroids
%calculates average centroid info
aveCentroidX = mean(centroid(:,1));
aveCentroidY = mean(centroid(:,2));

for i = 1:length(centroid)
    centDiffX = aveCentroidX - centroid(i,1);
    centDiffY = aveCentroidY - centroid(i,2);
    act = [];
    act = imtranslate(rotImNormPad{i}, [centDiffX, centDiffY]);
    rotImNormTrans{i} = act;
end

montage(rotImNormTrans);
