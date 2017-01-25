%BLOCK SHADING CORRECTION SCRIPT
%Does shading correction using the given algorithm, using the mean white
%and mean black image. Plots the scan and shading corrected scan. Also
%plots the gradient of target greyscale vs reference greyscale in shading
%correction.

clc;
clearvars;
close all;

%instantise an object which loads the data
block_data = BlockData_140316('../data/140316');

%plot the sample scan image
imagesc_truncate(double(block_data.loadSample(1)));
colorbar;
colormap gray;

%set up shading correction for the data set
block_data.addShadingCorrector(@ShadingCorrector,false);

%plot the shading corrected sample scan image
imagesc_truncate(double(block_data.loadSample(1)));
colorbar;
colormap gray;

%plot the gradient in the shading correction
imagesc_truncate(block_data.shading_corrector.b_array);
colorbar;