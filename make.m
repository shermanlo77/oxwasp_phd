clc;
clearvars;
close all;

disp('Chapter 3 - Literature Review'); 
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

disp('Chapter 6 - Compound Poisson');
inferenceIntro;
inferenceSubsample;
this = BandwidthSelection();
this.run();
this.printResults();
this = BandwidthSelection2();
this.run();
this.printResults();
nulliid;
allNullScript;
allNull;
defectExample;
defectSimulation;
defectDetect;

disp('Front cover');
frontCover;
