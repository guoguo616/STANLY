%% register tiff files to each other
% reg = imregister(test,target, 'translation', opt, met);

[opt,met] = imregconfig('monomodal');

target = sortIm{1};
reg{1} = sortIm{1};
for i = 2:length(sortIm)
    moving = sortIm{i};
    temp = imregister(moving,target, 'translation', opt, met);
    reg{i} = temp;
    clear temp
end