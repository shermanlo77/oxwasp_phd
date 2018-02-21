%PLOT MEAN VAR FIT
%Plot all the data's mean and variance frequency density heatmap for each type of shading correction
%Also fits and plots all regressions

this = Experiment_GlmMse_Mar16();
this.plotFullFit();

this = Experiment_GlmMse_July16_30deg();
this.plotFullFit();

this = Experiment_GlmMse_July16_120deg();
this.plotFullFit();

this = Experiment_GlmMse_Sep16_30deg();
this.plotFullFit();

this = Experiment_GlmMse_Sep16_120deg();
this.plotFullFit();