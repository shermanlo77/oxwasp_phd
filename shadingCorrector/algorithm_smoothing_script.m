%MEAN VARIANCE RELATIONSHIP BEFORE/AFTER SHADING CORRECTION SCRIPT
%Plots the sample mean and sample variance frequency density heat map
%before and after shading correction. A number of shading correction were
%considered:
    %unsmoothed b/w
    %unsmoothed b/g/w
    %polynomial smoothed order 2/2/2 on b/g/w

clc;
clearvars;
close all;

rng(uint32(364937754), 'twister');


%number of bins in the frequency density heatmap
n_bin = 100;

n_repeat = 20;

%instantise an object which loads the data
block_data = BlockData_140316('../data/140316');

shadingCorrection_ANOVA(block_data, 10, @ShadingCorrector_mean, 1, [3,3,3], n_repeat);


shadingCorrection_ANOVA(block_data, 10, @ShadingCorrector_median, 1, [3,3,3], n_repeat);


shadingCorrection_ANOVA(block_data, 10, @ShadingCorrector_gaussian, 1, [0.1,0.1,0.1], n_repeat);
