%SCRIPT: DEFECT LINE
%Plot Gaussian image with line defect with plane contimation
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

randStream = RandStream('mt19937ar','Seed',uint32(2816384857)); %instantise a rng

imageSize = 256;
radius = 20; %kernel radius
trueNullStd = 2; %multiplier in the contimation
trueNullMeanGrad = [0.01, 0.01]; %gradient of contimation
%distribution parameters of the alt distribution
altMean = 2;
altStd = 1;

defectSimulator = PlaneMultLine(randStream, trueNullMeanGrad, trueNullStd, altMean, altStd);
defectExample(defectSimulator, imageSize, radius);