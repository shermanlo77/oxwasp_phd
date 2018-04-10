addpath(genpath(fullfile('loadData')));
addpath(genpath(fullfile('compoundPoisson')));
addpath(genpath(fullfile('regressor')));
addpath(genpath(fullfile('shadingCorrector')));
addpath(genpath(fullfile('meanVar')));
addpath(genpath(fullfile('inference')));

% this = Experiment_ReferenceShadingCorrection_Mar16();
% this.run();
% 
% this = Experiment_ReferenceShadingCorrection_July16();
% this.run();
% 
% this = Experiment_ReferenceShadingCorrection_Sep16();
% this.run();

this = Experiment_MeanVarFit_Mar16();
this.run();
this.printResults();

this = Experiment_MeanVarFit_July16_30deg();
this.run();
this.printResults();

this = Experiment_MeanVarFit_July16_120deg();
this.run();
this.printResults();

this = Experiment_MeanVarFit_Sep16_30deg();
this.run();
this.printResults();

this = Experiment_MeanVarFit_Sep16_120deg();
this.run();
this.printResults();


this = Experiment_GlmMse_Mar16();
this.run();
this.printResults();

this = Experiment_GlmMse_July16_30deg();
this.run();
this.printResults();

this = Experiment_GlmMse_July16_120deg();
this.run();
this.printResults();

this = Experiment_GlmMse_Sep16_30deg();
this.run();
this.printResults();

this = Experiment_GlmMse_Sep16_120deg();
this.run();
this.printResults();


this = Experiment_GlmDeviance_Mar16();
this.run();
this.printResults();

this = Experiment_GlmDeviance_July16_30deg();
this.run();
this.printResults();

this = Experiment_GlmDeviance_July16_120deg();
this.run();
this.printResults();

this = Experiment_GlmDeviance_Sep16_30deg();
this.run();
this.printResults();

this = Experiment_GlmDeviance_Sep16_120deg();
this.run();
this.printResults();


this = Experiment_ZNull();
this.run();
this.printResults();

this = Experiment_ZNull_mse();
this.run();
this.printResults();

this = Experiment_NoDefect_Plane();
this.run();
this.printResults();

this = Experiment_NoDefect_Sinusoid();
this.run();
this.printResults();

this = Experiment_SimulateRoc_Squares();
this.run();
this.printResults();

this = Experiment_SimulateRoc_Line();
this.run();
this.printResults();
