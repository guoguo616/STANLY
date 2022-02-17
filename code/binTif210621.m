%% extracting multiple images from composite histological images
% import template images

close all; clear; clc
addpath('func/readObj/');
heData=fullfile('./data/Visium_HE_sections/'); % location of tifs to be aligned
templateData = fullfile('./data/templates/'); % location of template
templateImage = imread(fullfile(templateData,'tinyimage.jpg')); % template for fiducials
allFiles = dir(fullfile(heData,'*.tif'));
numTifs=numel(allFiles);       %counts number of directories in data folder
allFiles=struct2table(allFiles);  %convert struct to table
brainObj = readObj(fullfile(templateData,'brain-outline.obj'));

scale=0.1;  % amount the images will be scaled for alignment
d3=(6.5/9000) * scale;  %mm/pixel resolution of template image, 9000 pixels between border, scaled to data
imLocEst = [43,40,46,44,45,46,45,45,45,43,43,44,45,44,46,46];   % approximate paxinos slide according to Yann
[imSort,imSortIdx] = sort(imLocEst,"ascend");
figure; subplot(2,2,1); imshow(templateImage); title("Original Template"); subplot(2,2,2); imshow(templateImage(:,:,1)); title("Red Channel"); subplot(2,2,3); imshow(templateImage(:,:,2));  title("Green Channel"); subplot(2,2,4); imshow(templateImage(:,:,3)); title("Blue Channel");
thickness = 0.001;  % mm thickness of each slice, i.e. 0.001 = 1 micron
%% assign corner features
% locations of each corner fiducial, both full size and scaled, defined by Zeru
upperLeftLoc = templateImage(720:1319,530:1229,:); ul = imresize(upperLeftLoc, scale);
upperRightLoc = templateImage(720:1319,10350:11049,:); ur = imresize(upperRightLoc, scale);
lowerLeftLoc = templateImage(10750:11349, 530:1229,:); ll = imresize(lowerLeftLoc, scale);
lowerRightLoc = templateImage(10750:11349, 10350:11049,:); lr = imresize(lowerRightLoc, scale);
templateImage = imresize(templateImage, scale);
%% detect and extract features

% upperLeftPoints = detectBRISKFeatures(rgb2gray(ul));
% upperRightPoints = detectBRISKFeatures(rgb2gray(ur));
% lowerLeftPoints = detectBRISKFeatures(rgb2gray(ll));
% lowerRightPoints = detectBRISKFeatures(rgb2gray(lr));

% figure; subplot(2,2,1); imshow(ul); hold on; plot(upperLeftPoints); subplot(2,2,2); imshow(ur); hold on; plot(upperRightPoints); subplot(2,2,3); imshow(ll); hold on; plot(lowerLeftPoints); subplot(2,2,4); imshow(lr); hold on; plot(lowerRightPoints);

% [upperLeftFeat, upperLeftPoints] = extractFeatures(rgb2gray(upperLeftLoc), upperLeftPoints);

%% extracting HE information

for i=1:numTifs
    filename=cell2mat(fullfile(allFiles.folder(i),allFiles.name(i)));
    I = imread(filename);
    v = size(templateImage, 1)/size(I,1);   % calculates an image specific constant to resize to the template
    ir = imresize(I,v);
    im{i}=ir;
    clear I;
end
gIm=cellfun(@rgb2gray,im,'UniformOutput',false);

%% locate fiducials and rotate images into proper orientation

% prepare template

gTem = rgb2gray(templateImage);
bin = ~imbinarize(gTem); bin = imfill(bin, 'holes'); % binarize the template and fill edge dots
binF = bwareafilt(bin, [0,1000]); binF = double(binF);  % filter out the brain section so fiducials are the regions used for alignment

%% prepare fiducials
ulg = rgb2gray(ul); ulbin = ~imbinarize(ulg); ulf = double(imfill(ulbin,'holes'));
urg = rgb2gray(ur); urbin = ~imbinarize(urg); urf = double(imfill(urbin,'holes'));
llg = rgb2gray(ll); llbin = ~imbinarize(llg); llf = double(imfill(llbin,'holes'));
lrg = rgb2gray(lr); lrbin = ~imbinarize(lrg); lrf = double(imfill(lrbin,'holes'));
figure; subplot(2,2,1); imshow(ulf); subplot(2,2,2); imshow(urf); subplot(2,2,3); imshow(llf); subplot(2,2,4); imshow(lrf);

%% prepare HE images

for i = 1:length(gIm)

    tem = ~imbinarize(gIm{i}); tem = imfill(tem,'holes');   % fills dots of border
    imBord = bwareafilt(tem, [0,1000]); % remove the tissue slice to search for fiducials
    slice = bwareafilt(tem, 1);
    bord{i} = double(imBord);   % extracted borders of each image
    sliceMask{i} = double(slice);
end

%% find fiducials
% finds proper orientation of images in relation to the template
aR = [0, 90, 180, 270];
for i = 1:length(bord)
    n = 1;
    for r = 0:90:270
        act = imrotate(bord{i}, r);
        cul = normxcorr2(ulf, act);% searches for fiducials in each orientation
        C(1, n) = max(cul(:));  % extracts max match for each fiducial
        cur = normxcorr2(urf, act);
        C(2, n) = max(cur(:));
        cll = normxcorr2(llf, act);
        C(3, n) = max(cll(:));
        clr = normxcorr2(lrf, act);
        C(4, n) = max(clr(:));

        n = n + 1;
    end
    C = mean(C);    % finds the mean matching of all fiducials per orientation
    [mc, midx] = max(C);    % finds the best fit rotation
    rotBord{i} = imrotate(bord{i}, aR(midx));   % performs proper rotation of border image
    rotIm{i} = imrotate(gIm{i}, aR(midx));  % performs proper rotation of grayscale image
    rotSliceMask{i} = imrotate(sliceMask{i}, aR(midx));  % performs proper rotation of slice image
end

figure; montage(rotBord)

clear act bord 
%% min-max normalization
% normalizes the images to themselves for more even values
for i = 1:length(rotIm)
    act = [];
    act = im2double(rotIm{i});
    act = (act - min(act(:))) / (max(act(:)) - min(act(:)));
    act = act .* rotSliceMask{i};
    rotImNorm{i} = act;
end
figure; montage(rotImNorm)

%% remove junk

% se = strel('disk',4);
% ie = imerode(rotImNorm{3},se);

%% put images into "proper" order

for i = 1:length(rotImNorm)
    sortIm{i} = rotImNorm{imSortIdx(i)};
end

%% register tiff files to each other
% reg = imregister(test,target, 'translation', opt, met);

[opt,met] = imregconfig('monomodal');

%target = sortIm{1};
reg{1} = sortIm{1};
for i = 2:length(sortIm)
    n = i-1;
    target = reg{n};
    moving = sortIm{i};
    temp = imregister(moving,target, 'translation', opt, met);
    reg{i} = temp;
    clear temp
end

for i = 2:length(reg)
    moving = reg{i};
    temp = imregister(moving,target, 'affine', opt, met);
    affReg{i} = temp;
    clear temp
end
%% match binarized image sizes to be combined

for i = 1:length(reg)
    Vbw(:,:,i)=reg{i};     %size corrected images concatenated into a 3d volume
end

niftiwrite(Vbw,'testReg210621.nii');
vInfo = niftiinfo('testReg210621.nii');
vInfo.PixelDimensions = [d3, d3, thickness];
niftiwrite(Vbw,'testReg210621.nii',vInfo);

for i = 1:length(reg)
    Vba(:,:,i)=affReg{i};     %size corrected images concatenated into a 3d volume
end

niftiwrite(Vba,'testAffReg210621.nii');
vInfo = niftiinfo('testAffReg210621.nii');
vInfo.PixelDimensions = [d3, d3, thickness];
niftiwrite(Vba,'testAffReg210621.nii',vInfo);

% [s,i]=cellfun(@size,sortIm);
% maxh=max(s);                    %find maximum pixel height of images
% maxw=max(i);                    %fine maximum pixel width of images
% sortImPad = sortIm;
% for i=1:length(sortImPad)
% 
%     h=size(sortImPad{i},1);
%     w=size(sortImPad{i},2);
%     if h < maxh
%       hd=maxh-h;
%       hpa=floor(hd/2);
%       sortImPad{i}=padarray(sortImPad{i},[hpa,0],0,'both');
%       if size(sortImPad{i},1)==(maxh-1)
%           sortImPad{i}=padarray(sortImPad{i},[1,0],0,'pre');
%       else
%       end
%     else
%         sortImPad{i}=sortImPad{i};
%     end
%     if w < maxw
%         wd=maxw-w;
%         wpa=floor(wd/2);
%         sortImPad{i}=padarray(sortImPad{i},[0,wpa],0,'both');
%         if size(sortImPad{i},2)==(maxw-1)
%             sortImPad{i}=padarray(sortImPad{i},[0,1],0,'pre');
%         else
%         end
%     else
%         sortImPad{i}=sortImPad{i};     %size corrected images contained in cell
%     end
% end

%%
% se=strel('disk',5);
% %I=imread('example_image.png');  %import image
% for i=1:length(gIm)
%     mask=ones(size(gIm{i}));             %create empty mask for active contour
%     bw{i}=activecontour(gIm{i},mask,300);
% %     bw2=bwareafilt(bw,[10000,10000000]);     %filter to only include large blobs
% %     Ie=imerode(bw2,se);             %erode to remove overlap between blobs
% %     Id=imdilate(Ie,se);             %bring back to original size
% %     Io=bwareaopen(Id,10000);        %remove extra blobs
% %     Il=bwlabel(Io);                 %turn binarized image into a label
% %     Ic=label2rgb(Il);               %colorize labeled image
% end

%% match binarized image sizes to be combined
% [s,i]=cellfun(@size,bw);
% maxh=max(s);                    %find maximum pixel height of images
% maxw=max(i);                    %fine maximum pixel width of images
% gIm=bw;
% gn = g;
% for i=1:length(gIm)
%     h=size(gIm{i},1);
%     w=size(gIm{i},2);
%     if h < maxh
%       hd=maxh-h;
%       hpa=floor(hd/2);
%       gIm{i}=padarray(gIm{i},[hpa,0],0,'both');
%       gn{i}=padarray(gn{i},[hpa,0],0,'both');
%       if size(gIm{i},1)==(maxh-1)
%           gIm{i}=padarray(gIm{i},[1,0],0,'pre');
%           gn{i}=padarray(gn{i},[1,0],0,'pre');
%       else
%       end
%     else
%         gIm{i}=gIm{i};
%         gn{i} = gn{i};
%     end
%     if w < maxw
%         wd=maxw-w;
%         wpa=floor(wd/2);
%         gIm{i}=padarray(gIm{i},[0,wpa],0,'both');
%         gn{i}=padarray(gn{i},[0,wpa],0,'both');
%         if size(gIm{i},2)==(maxw-1)
%             gIm{i}=padarray(gIm{i},[0,1],0,'pre');
%             gn{i}=padarray(gn{i},[0,1],0,'pre');
%         else
%         end
%     else
%         gIm{i}=gIm{i};     %size corrected images contained in cell
%         gn{i} = gn{i};
%     end
%     Vbw(:,:,i)=gIm{i};     %size corrected images concatenated into a 3d volume
%     Vg(:,:,i) = gn{i};
% end
% 
% %% create initial 3d image of mask and slices
% Vb=mat2gray(Vbw);
% niftiwrite(Vb,'binarized.nii');
% Vbinfo=niftiinfo('binarized.nii');
% x=d3*(1/scale);
% Vbinfo.PixelDimensions=[0.012 0.012 x];
% niftiwrite(Vb,'binarized.nii',Vbinfo);
