%SCRIPT: DEFECT DUST
%Plot Gaussian image with dust defect with plane contimation
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

directory = fullfile('reports','figures','inference');
imageSize = 256;
%distribution parameters of the alt distribution
altMean = 3;
altStd = 1;
trueNullStd = 2; %multiplier in the contimation
trueNullMeanGrad = [0.01, 0.01]; %gradient of contimation


%=====DUST DEFECT=====%
prefix = 'defectDustExample';
randStream = RandStream('mt19937ar','Seed',uint32(676943031)); %instantise a rng
radius = 20; %kernel radius
altP = 0.1; %proportion of image defected
defectSimulator = PlaneMultDust(randStream, trueNullMeanGrad, trueNullStd, altP, altMean, altStd);
examplePlot = DefectExample(randStream);
examplePlot.setPlotDefectNullStd(true);
examplePlot.plotExample(defectSimulator, imageSize, radius, directory, prefix);

%=====LINE DEFECT=====%
prefix = 'defectLineExample';
randStream = RandStream('mt19937ar','Seed',uint32(2816384857)); %instantise a rng
radius = 20; %kernel radius
lineThickness = 5;
defectSimulator = PlaneMultLine(randStream, trueNullMeanGrad, trueNullStd, altMean, altStd, ...
    lineThickness);
examplePlot = DefectExample(randStream);
examplePlot.setPlotDefectNullStd(true);
examplePlot.plotExample(defectSimulator, imageSize, radius, directory, prefix);

%=====SQUARE DEFECT (SMALL KERNEL)=====%
prefix = 'defectSquareExample';
randStream = RandStream('mt19937ar','Seed',uint32(4120988592)); %instantise a rng
radius = 20; %kernel radius
defectSize = 30; %size of the square defect
defectSimulator = PlaneMultSquare(randStream, trueNullMeanGrad, trueNullStd, defectSize, ...
    altMean, altStd);
examplePlot = DefectExample(randStream);
examplePlot.setNInitial(10);
examplePlot.setCLimNullMean([-2.5, 5]);
examplePlot.plotExample(defectSimulator, imageSize, radius, directory, prefix);

%=====SQUARE DEFECT (LARGE KERNEL)=====%
prefix = 'defectSquare2Example';
randStream = RandStream('mt19937ar','Seed',uint32(4120988592)); %instantise a rng
radius = 40; %kernel radius
defectSize = 30; %size of the square defect
defectSimulator = PlaneMultSquare(randStream, trueNullMeanGrad, trueNullStd, defectSize, ...
    altMean, altStd);
examplePlot = DefectExample(randStream);
examplePlot.setNInitial(10);
examplePlot.setCLimNullMean([-2.5, 5]);
examplePlot.plotExample(defectSimulator, imageSize, radius, directory, prefix);