%% create a function that does the lesion extraction with two input images
function [segVol] = kmeansSegmentation(volLoc)
vol = double(niftiread(volLoc));
volInfo = niftiinfo(volLoc);
%vol(vol==0) = NaN;
origSize = size(vol);

%% normalize images for comparison

% use min-max normalization to bring both images into 0-1 scale
volNorm = (vol - min(min(min(vol)))) ./ (max(max(max(vol))) - min(min(min(vol))));
% figure; montage(volNorm(:,:,29:100));
% title('Pre-op MRI');

% consider a varargin for sigma
sigma = 2;
volGaus = imgaussfilt3(volNorm, sigma);

%% run k-means to segment images

flatPreopNorm = reshape(volGaus, 1, []);

% consider a varargin for k-means
k = 4;
preopIdx = kmeans(flatPreopNorm', k);

segVol = single(reshape(preopIdx, origSize));
figure
montage(segVol(:,:,:)); caxis([0 k]);
end