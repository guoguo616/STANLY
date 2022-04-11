%% take aligned images and generate volume
close all
addpath('./func/')
imLocEst = [43,40,46,44,45,46,45,45,45,43,43,44,45,44,46,46];   % approximate paxinos slide according to Yann
[imSort,imSortIdx] = sort(imLocEst,"ascend");

% create a volume based on expected order above. assumes slices are
% sequential, so no space between them currently
for i = 1:16
    vol{i} = rotImNormTrans{imSortIdx(i)};
end

%% register image files to each other
% reg = imregister(test,target, 'translation', opt, met);

[opt,met] = imregconfig('monomodal');
% opt.GradientMagnitudeTolerance = 1.000e-02;
% opt.MinimumStepLength = 1.000e-01;
% opt.MaximumStepLength = 6.25e-02;
% opt.MaximumIterations = 300;
% opt.RelaxationFactor = 5.000e-01;

reg{1} = vol{1};
% this should be unnecessary now that the centroid alignment has been added

% reg{1} = vol{1};
% for i = 2:length(vol)
%     n = i-1;
%    target = reg{n};
%     moving = vol{i};
%     temp = imregister(moving,target, 'translation', opt, met);
%     reg{i} = temp;
%     clear temp
% end

for i = 2:length(vol)
    n = i-1;
    target = reg{n};
    moving = vol{i};
    temp = imregister(moving,target, 'affine', opt, met);
    reg{i} = temp;
    clear temp
end


figure; montage(vol)
figure; montage(reg)


% niftiwrite(vol, 'testSorted.nii');
% niiInfo = niftiinfo('testSorted.nii');
% niiInfo.PixelDimensions = [pixDim, pixDim, sliceThickness];
% niftiwrite(vol, 'testSorted.nii', niiInfo);


%% testing segmentation of volume
% 
% segVol = kmeansSegmentation(fullfile('testSorted.nii'));