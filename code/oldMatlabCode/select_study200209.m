function [probe, gene, gene_id, F] = select_study(donor_name)

    DATA_DIR = 'data';
    CACHE_DIR = '_cache';


    % Transformation matrix from MNI to pixel space
%     T = ...
%     [1 0 0 -96;...
%     0 1 0 -132;...
%     0 0 1 -78;...
%     0 0 0 1];

    % Read image
    info = niftiinfo('./data/current_brainmask_20200211.nii.gz');
    image = niftiread(info);

    T = info.Transform.T';

    if strcmp(donor_name, 'all_donors')
        fprintf('All donors selected.\n');

        % Read gene expression data
        gene_exp = [];
        probes = [];
        filelist = dir(fullfile(DATA_DIR, '*.mat'));
        for i=1:length(filelist)
            fprintf(['Reading ', filelist(i).name, sprintf('\t(%d/%d)\t........... ', i, length(filelist))]);
            load(fullfile(filelist(i).folder, filelist(i).name));
            gene_exp{i} = X;
            probes{i} = V;
            fprintf('DONE\n');
        end

        % merge all the probes
        probe = [];
        gene = [];
        gene_id = GN;
        for i = 1:length(filelist)
            probe = [probe; probes{i}'];
            gene = [gene; gene_exp{i}'];
        end

        cache_path = fullfile(CACHE_DIR, 'gpdist_all_donors.mat');
    elseif strcmp(donor_name, 'whole_brain_donors')
        fprintf('Whole Brain Donors selected.\n');
        % Read gene expression data
        gene_exp = [];
        probes = [];
        filelist = dir(fullfile(DATA_DIR, '*.mat'));
        for i=1:5:6
            fprintf(['Reading ', filelist(i).name, sprintf('\t(%d/%d)\t........... ', i, length(filelist))]);
            load(fullfile(filelist(i).folder, filelist(i).name));
            gene_exp{i} = X;
            probes{i} = V;
            fprintf('DONE\n');
        end

        % merge all the probes
        probe = [];
        gene = [];
        gene_id = GN;
        for i = 1:5:6
            probe = [probe; probes{i}'];
            gene = [gene; gene_exp{i}'];
        end
        cache_path = fullfile(CACHE_DIR, 'gpdist_whole_brain_donors.mat');
    else
        switch(donor_name)
            case 'donor9861'
                fprintf('Donor 9861 selected.\n');
            case 'donor10021'
                fprintf('Donor 10021 selected.\n');
            case 'donor12876'
                fprintf('Donor 12876 selected.\n');
            case 'donor14380'
                fprintf('Donor 14380 selected.\n');
            case 'donor15496'
                fprintf('Donor 15496 selected.\n');
            case 'donor15697'
                fprintf('Donor 15697 selected.\n');
            otherwise
                error(['Donor name "', donor_name, '" not recognized.']);
        end

        filename = ['MicroarrayExpression_SampleAnnot_', donor_name, '.mat'];

        fprintf(['Reading ', filename, '\t........... ']);
        load(fullfile(DATA_DIR, filename));
        gene = X';
        probe = V';
        gene_id = GN;
        fprintf('DONE\n');

        cache_path = fullfile(CACHE_DIR, ['gpdist_', donor_name, '.mat']);
    end

    % convert from MNI coordinates to pixel coordinates
    probe = mni2cor(probe,T);

    if exist(cache_path, 'file')
        fprintf(['Reading cached geodesic distances for ', donor_name]);
        load(cache_path, 'F');
        fprintf('.......................DONE\n');
    else
        % compute the geodesic distance matrix among probes
        tic
        F = gpdist(image, probe);
        toc

        % Cache the result for future use
        save(cache_path, 'F');
    end
end
