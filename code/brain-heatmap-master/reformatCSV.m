%% remake Mouse_values.csv to work with heatmap script
clear; close all; clc
tableToChange = readtable('./ontology/Mouse_values.csv');
id = [];
acronym = [];
name = [];
value = [];
n = 1;
for i = 1:height(tableToChange)
    if any(tableToChange.value(i))
        id(n) = tableToChange.id(i);
        acronym{n} = tableToChange.acronym(i);
        name{n} = tableToChange.acronym(i);
        value(n) = tableToChange.value(i);
        n = n + 1;
    else
        continue
    end
end

id = id'; acronym = acronym'; name = name'; value = value';

newTable = table(id, acronym, name, value);

writetable(newTable, 'MalesAutismDEGBrainRegions_updated.xlsx');