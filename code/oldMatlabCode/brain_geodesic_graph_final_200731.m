%% CONFIG
clear
close all
clc

addpath('../../build');

%% SELECT STUDY & LOAD UP THE DATA

[probe, gene, gene_id, F] = select_study_leftH200130('all_donors'); filename = 'donorAllLeft210823.nii';
% [probe, gene, gene_id, F] = select_study_leftH200130('all_donors'); filename = 'donorAllLabelsLeftHemi200224.nii'
% [probe, gene, gene_id, F] = select_study200128('whole_brain_donors'); filename = 'donorWholeBrainsLabelsNoBounds200218.nii';
% [probe, gene, F] = select_study('donor9861');
% [probe, gene, F] = select_study('donor10021');
% [probe, gene, F] = select_study('donor12876');
% [probe, gene, F] = select_study('donor14380');
% [probe, gene, F] = select_study('donor15496');
% [probe, gene, F] = select_study('donor15697');

%% CONSTRUCT A GRAPH BASED ON THE GEODESIC DISTANCES
X = probe';
GE = gene';

% we will connect up to k neighbors
k = 12;  
[sorted_F, ind] = sort(F,2);
nn = ind(:, 2:k+1);
nn_dist = sorted_F(:, 2:k+1);

% connect neighbors and build edges 'E'
[I,J] = find(~isinf(nn_dist) & nn_dist > 0);
E = [I, nn(sub2ind(size(nn), I, J))];
E = sort(E, 2);
E = unique(E, 'rows')';

% Plot
MARKER_SIZE = 30; % size of the node to display
figure; scatter3(X(1,:), X(2,:), X(3,:), MARKER_SIZE, 'k', 'filled'); % plot points
axis equal
hold on
plot3([X(1,E(1,:)); X(1,E(2,:))],...
      [X(2,E(1,:)); X(2,E(2,:))],...
      [X(3,E(1,:)); X(3,E(2,:))],...
      'Color', [0.7,0.7,0.7]);
title('Probes');


%% SPECTRAL BASIS DECOMPOSITION USING GENE EXPRESSIONS
small = 1e-6;  % infinitesimal number
N = size(X,2);

% Weighted adjacency matrix
I = [E(1,:), E(2,:)];
J = [E(2,:), E(1,:)];
V = sum(GE(I,:).*GE(J,:), 2) ./ (sum(GE(I,:).^2,2).*sum(GE(J,:).^2,2));   %1./sum((GE(I,:) - GE(J,:)).^2 + small, 2);
V = (V - min(V))/(max(V)-min(V));
W = sparse(I,J,V, N,N);% + A;

% Weighted Laplacian
D = sum(W,2);
isolated = find(D==0);
connected = setdiff(1:N, isolated);
% D(find(D==0)) = small;
D = diag(D);
L = D-W;

% Spectral embedding
[evec, eval] = eigs(L(connected, connected), 300, small);
[eval, idx] = sort(real(diag(eval)));
evec = real(evec(:, idx))./sqrt(eval');
idx = find(eval < sqrt(small));
evec(:,idx) = [];
eval(idx) = [];

%% SPECTRAL CLUSTERING
n_cluster = 81; % specify the number of clusters

clusters = kmeans(evec, n_cluster);     % kmeans returns visually better results
% clusters = dbscan(evec, 0.18, 4)+2;        % dbscan seems to be more rigorous


figure;
% plot3([X(1,E(1,:)); X(1,E(2,:))],...
%       [X(2,E(1,:)); X(2,E(2,:))],...
%       [X(3,E(1,:)); X(3,E(2,:))],...
%       'Color', [0.7,0.7,0.7]);
% hold on
clr = rand(n_cluster,3);
clr(1, :) = [0.7,0.7,0.7];   % in case of DBSCAN gray probes are unclassified (i.e. uncertain to be added to a cluster)
scatter3(X(1,connected), X(2,connected), X(3,connected), MARKER_SIZE*2, clr(clusters, :), 'filled'); % plot points
axis equal
title('Spectral Clustering using Weighted Laplacian');

%%
nmembers = [];
for i=1:n_cluster
    nmembers = [nmembers, sum(clusters==i)];
end
figure; bar(nmembers); title('Number of Genes per Cluster');


%% SILHOUETTE METHOD
% avg_s = [];
% krange = 1:10:1000;
% for n_cluster = krange
%     i = find(krange == n_cluster);
%     n_cluster
%     clusters = kmeans(evec, n_cluster);
%     clusters_all(:,i) = clusters;
%     s = silhouette(evec, clusters);
%     avg_s = [avg_s; mean(s)];
% end
% figure; plot(krange, avg_s);

%% Visualization
% Read image

% info = niftiinfo('./data/leftHemMask.nii.gz');
% image = niftiread(info);
% nInfo = info;
% nInfo.Datatype = 'double';
% 
% 
% for i = 1:length(clusters_all)
%     active_clustering = clusters_all{i};
%     n_cluster = length(unique(active_clustering));
%     labels = zeros(size(image));
%     for j=1:n_cluster
%         fprintf('%d\n', j)
%         bl = zeros(size(image));
%         B = highlightCluster(image, probe(connected, :), probe(connected(active_clustering==j), :));
% %         labels_sep{j}=bl+B;
%         labels = labels + j*B;
%         b = zeros(size(image));
%         b(labels>j)=1;
%         labels(b == 1) = 0;
%     end
%     formatSpec='left_kmeans_%d.nii';
%     str=sprintf(formatSpec,n_cluster);
%      niftiwrite(labels,str,nInfo);
% end

%% jaccard testing
% for n = 1:length(labels_all)
%     active_c = reshape(labels_all{n},1,[]);
%     for m = 1:length(labels_all)
%         test_c = reshape(labels_all{m},1,[]);
%         intersection = sum(active_c & test_c);
%         union = sum(active_c | test_c);
%         init(m) = intersection./union;
%     end
%     overlap_j{n} = init;
% end
