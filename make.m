%MIT License
%Copyright (c) 2019 Sherman Lo

clc;
clearvars;
close all;

disp('Chapter 3 - Data Collection'); 
plotScans;
oddEvenPlot;
shadingCorrectionExample;
this = ShadingCorrectionAnovaAbsNoFilter();
this.run();
this.printResults();
this = ShadingCorrectionAnovaAbsFilter();
this.run();
this.printResults();

disp('Chapter 4 - Compound Poisson');
cpHistogram;
this = CpEmAlgorithm();
this.run();
this.printResults();
cpLogLikelihoodPlot;

disp('Chapter 5 - Variance Prediction');
glmSelect; %parfor possible
varMeanExample;
varMeanCv; %parfor possible
devianceGraph;
varMeanResidual;

disp('Chapter 6 - Inference');
inferenceIntro;
inferenceSubsample;
this = BandwidthSelection();
this.run();
this.printResults();
this = BandwidthSelection2();
this.run();
this.printResults();
nullIid;
allNullScript;
allNull;
defectExample;
defectSimulation;
defectDetect;
inferenceSubRoi;

disp('Front cover');
frontCover;
