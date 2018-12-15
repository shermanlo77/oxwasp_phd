%SCRIPT: DEFECT SQUARE
%Using non-sensible kernel size
%Plot Gaussian image with square defect with plane contimation
%Plot empirical null mean
%Plot empirical null std
%Plot roc curve before and after contimation
%For the z_alpha=2 level, print:
  %type 1 error
  %type 2 error
  %area under ROC

clc;
clearvars;
close all;

directory = fullfile('reports','figures','inference','defectsimulation');
prefix = 'defectSquare';
randStream = RandStream('mt19937ar','Seed',uint32(4120988592)); %instantise a rng

imageSize = 256;
radius = 20; %kernel radius
trueNullStd = 2; %multiplier in the contimation
trueNullMeanGrad = [0.01, 0.01]; %gradient of contimation
defectSize = 30; %size of the square defect
%distribution parameters of the alt distribution
altMean = 3;
altStd = 1;

defectSimulator = PlaneMultSquare(randStream, trueNullMeanGrad, trueNullStd, defectSize, ...
    altMean, altStd);
%for a large altMean, the empirical null filter requires more initial values to jump to the other
%mode
defectExample = DefectExample();
defectExample.setNInitial(10);
defectExample.setCLimNullMean([-2.5, 5]);
defectExample.plotExample(defectSimulator, imageSize, radius, directory, prefix);