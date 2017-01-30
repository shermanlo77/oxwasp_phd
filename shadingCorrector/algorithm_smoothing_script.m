%ALGORITHM SMOOTHING SCRIPT
%Does ANOVA analysis on the shaded corrected b/g/w images using smoothing
%techniques on the training images: mean, median, gaussian filters.

clc;
clearvars;
close all;

%set random seed
rng(uint32(364937754), 'twister');


%number of bins in the frequency density heatmap
n_bin = 100;

n_repeat = 20;

%instantise an object which loads the data
block_data = BlockData_140316('../data/140316');

shadingCorrection_ANOVA(block_data, 10, @ShadingCorrector_mean, 1, [3,3,3], n_repeat);


shadingCorrection_ANOVA(block_data, 10, @ShadingCorrector_median, 1, [3,3,3], n_repeat);


shadingCorrection_ANOVA(block_data, 10, @ShadingCorrector_gaussian, 1, [0.1,0.1,0.1], n_repeat);
