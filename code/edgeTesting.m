%% testing edge extraction

I = rotImNormTrans{9};
[Igrad, Idir] = imgradient(I);
Iedge = edge(I);

Inan = zeros(size(I));
Inan(I == 0) = NaN;
Inan(I > 0) = I(I>0);

% Imean = mean(mean(Inan,'omitnan'),'omitnan');
% Istd = std(std(Inan,'omitnan'),'omitnan');
% 
% Iz = zeros(size(I));
% 
% for i = 1:size(I, 1)
%     for j = 1:size(I,2)
%         if isnan(Inan(i,j))
%             Iz(i,j) = NaN;
%         else
%             Iz(i,j) = (Inan(i,j) - Imean) / Istd;
%         end
%     end
% end
% 
% x = (Inan - Imean) / Istd;
% iter = 40;
% startingSigma = 0.1;
% sI = zeros(size(I));
% for i = 1:iter
%     sigma = startingSigma * i;
%     gI{i} = imgradient(imgaussfilt(I, sigma));
%     sI = sI + gI{i};
% end
% 
% sI = sI./iter;
% imshow(sI)


%%
for i = 1:length(rotImNormTrans)
    gI{i} = imgradient(imgaussfilt(rotImNormTrans{i}, 2));
    gIt{i} = gI{i}(gI{i} > 0.15);
end