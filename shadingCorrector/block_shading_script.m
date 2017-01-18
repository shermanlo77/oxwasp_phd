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

%declare an array reference_stack which stores the mean black and mean
%white image
reference_stack = zeros(block_data.height,block_data.width,2);
%load and save the mean black image
reference_stack(:,:,1) = mean(block_data.loadBlackStack(),3);
%load and save the mean white image
reference_stack(:,:,2) = mean(block_data.loadWhiteStack(),3);

%instantise a shading corrector and set it up using the reference images
shading_corrector = ShadingCorrector(reference_stack);
shading_corrector.calibrate();

%get the first scan image of the sample
sample_scan = double(block_data.loadSample(1));

%plot the sample scan image
imagesc_truncate(sample_scan);
colorbar;
colormap gray;

%plot the shading corrected sample scan image
imagesc_truncate(shading_corrector.shadeCorrect(sample_scan));
colorbar;
colormap gray;

%plot the gradient in the shading correction
imagesc_truncate(shading_corrector.b_array);
colorbar;