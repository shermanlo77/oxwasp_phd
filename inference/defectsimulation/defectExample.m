%MIT License
%Copyright (c) 2019 Sherman Lo

%SCRIPT: DEFECT DUST
%Plot Gaussian image with defect with plane contamination
%Plot empirical null mean
%Plot empirical null std
%Plot roc curve before and after contamination

clearvars;
close all;

DebugPrint.newFile(mfilename);

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
examplePlot.setCLim([-4, 6]);
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
examplePlot.setCLim([-4, 6]);
examplePlot.plotExample(defectSimulator, imageSize, radius, directory, prefix);

DebugPrint.close();