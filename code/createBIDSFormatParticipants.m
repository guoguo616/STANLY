%% Template Matlab script to create an BIDS compatible participants.tsv file
% This example lists all required and optional fields.
% When adding additional metadata please use CamelCase
%
% DHermes, 2017
% updated by Zeru Peterson, 2021

%%
addpath(fullfile('/','home','zjpeters','matlabToolboxes','JSONio-main'));
clear;
root_dir = fullfile('/','home','zjpeters','Documents','otherLabs','visiumalignment','rawdata');
% project_label = 'templates';

participants_tsv_name = fullfile(root_dir, 'participants.tsv');

%% make a participants table and save

t = readtable('./sleepDepParticipants.xls');


writetable(t, participants_tsv_name, 'FileType', 'text', 'Delimiter', '\t');

%% associated data dictionary

template = struct( ...
                  'LongName', '', ...
                  'Description', '', ...
                  'Levels', [], ...
                  'Units', '', ...
                  'TermURL', '');



dd_json.sex = template;
dd_json.sex.Description = 'best match allen template slice';

%% Write JSON

json_options.indent = ' '; % this just makes the json file look prettier
% when opened in a text editor

jsonSaveDir = fileparts(participants_tsv_name);
if ~isdir(jsonSaveDir)
    fprintf('Warning: directory to save json file does not exist: %s \n', jsonSaveDir);
end

try
    jsonwrite(strrep(participants_tsv_name, '.tsv', '.json'), dd_json, json_options);
catch
    warning('%s\n%s\n%s\n%s', ...
            'Writing the JSON file seems to have failed.', ...
            'Make sure that the following library is in the matlab/octave path:', ...
            'https://github.com/gllmflndn/JSONio');
end
