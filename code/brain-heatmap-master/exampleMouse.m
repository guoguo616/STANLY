%% -----------------   Mouse   ---------------------------
clear; close all; clc
%target_region_acronym =  {'Isocortex','OLF','HIP','RHP','STR','PAL','HY','TH','MB','CB','P','MY'};

inputTable = readtable('MalesAutismDEGBrainRegions_updated.xlsx');
load('whiteToRedColormap.mat');

target_region_acronym = inputTable.acronym';
expressionLevel = inputTable.value';

% one dimensional vector containing "expression level" for target regions
% heatmap will be normalized to min and max of range
%expressionLevel = [14,1,4,4.5,3,11,3,8,2.3,6.4,3.4,5.5] %randn(1,12);    % [14,1,4,4.5,3,11,3,8,2.3,6.4,3.4,5.5]
% add color range
% colorscale = flipud(autumn);
% colorscale = colorscale(10:end-2,:);
colorscale = whiteToRedColormap;
target_region_colors = mapValuesToColorScale(expressionLevel, colorscale);
target_region_colors = colorTripletToHex(target_region_colors);
colorRegions(target_region_acronym, target_region_colors, 'ontology/Mouse_values.csv','mouse_output.csv');

! python2.7 colorize.py --inputfile mouse_output.csv --svgfile mouse_atlas_svg/mouseSagittal.svg > Mouseoutput.svg
! python2.7 colorize.py --inputfile mouse_output.csv --svgdirectory mouse_atlas_svg --outputfolder output_folder

imagesc(expressionLevel);
colormap(colorscale);
colorbar();
ax = gca;
ax.XTickLabel = target_region_acronym;
%% -------------------------------- Mouse regions ------------------------------------
% copy paste the ontology tree from: http://atlas.brain-map.org/atlas#atlas=2&plate=100883921
% -----------------------------------------------------------------------------------   
% grey  Basic cell groups and regions
%  CH  Cerebrum
%  CTX  Cerebral cortex
%  CTXpl  Cortical plate
%  Isocortex  Isocortex
%  FRP  Frontal pole, cerebral cortex
%  AI  Agranular insular area
%  PTLp  Posterior parietal association areas
%  VISC  Visceral area
%  PL  Prelimbic area
%  SS  Somatosensory areas
%  TEa  Temporal association areas
%  VIS  Visual areas
%  ACA  Anterior cingulate area
%  ORB  Orbital area
%  RSP  Retrosplenial area
%  PERI  Perirhinal area
%  GU  Gustatory areas
%  MO  Somatomotor areas
%  ILA  Infralimbic area
%  ECT  Ectorhinal area
%  AUD  Auditory areas
%  OLF  Olfactory areas
%  HPF  Hippocampal formation
%  CTXsp  Cortical subplate
%  6b  Layer 6b, isocortex
%  CLA  Claustrum
%  EP  Endopiriform nucleus
%  LA  Lateral amygdalar nucleus
%  BLA  Basolateral amygdalar nucleus
%  BMA  Basomedial amygdalar nucleus
%  PA  Posterior amygdalar nucleus
%  CNU  Cerebral nuclei
%  STR  Striatum
%  PAL  Pallidum
%  BS  Brain stem
%  IB  Interbrain
%  TH  Thalamus
%  HY  Hypothalamus
%  MB  Midbrain
%  HB  Hindbrain
%  P  Pons
%  MY  Medulla
%  CB  Cerebellum
%  CBX  Cerebellar cortex
%  VERM  Vermal regions
%  HEM  Hemispheric regions
%  CBXmo  Cerebellar cortex, molecular layer
%  CBXpu  Cerebellar cortex, Purkinje layer
%  CBXgr  Cerebellar cortex, granular layer
%  CBN  Cerebellar nuclei
%  FN  Fastigial nucleus
%  IP  Interposed nucleus
%  DN  Dentate nucleus
