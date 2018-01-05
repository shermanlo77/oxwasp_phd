%BLOCK SHADING CORRECTION SCRIPT
%Does shading correction using the given algorithm, using the mean white, mean gray and mean black image.
%Plots the scan and shading corrected scan.
%Also plots the gradient of target greyscale vs reference greyscale in shading correction.

clc;
clearvars;
close all;

%instantise an object which loads the data
block_data = AbsBlock_July16_30deg();

fig = figure_latexSub;
imagesc(block_data.loadImage(1));
colormap gray;

block_data.addDefaultShadingCorrector();
%block_data.addShadingCorrector(ShadingCorrector(),1:5);

subplot(2,1,2);
imagesc(block_data.loadImage(1));
colormap gray;

bgw_data = Bgw_Mar16();

figure;
subplot(2,1,1,imagesc_truncate(bgw_data.reference_scan_array(3).loadImage(1)));
bgw_data.addShadingCorrector(ShadingCorrector_polynomial([2,2,2]),1:3);
subplot(2,1,2,imagesc_truncate(bgw_data.reference_scan_array(3).loadImage(1)));
