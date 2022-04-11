%% check HE alignment from 'Molecular Atlas of the Adult Mouse Brain'

heDir = fullfile('data','fromMolecularAtlas','HE');
heTable = struct2table(dir(fullfile(heDir, '*.jpg')));
scale = 0.1;
for i = 1:height(heTable)
    heImages{i} = imresize(imread(fullfile(heTable.folder{i},heTable.name{i})),scale);
end
montage(heImages)

%% begin processing

heGrey=cellfun(@rgb2gray,heImages,'UniformOutput',false);