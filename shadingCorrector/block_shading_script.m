%BLOCK SHADING CORRECTION SCRIPT
%Does shading correction using the given algorithm, using the mean white, mean gray and mean black image.
%Plots the scan and shading corrected scan.
%Also plots the gradient of target greyscale vs reference greyscale in shading correction.

clc;
clearvars;
close all;

%instantise an object which loads the data
block_data = AbsBlock_July16_30deg();

figure;
subplot(2,1,1);
imagesc(block_data.loadImage(1));
colormap gray;

block_data.addDefaultShadingCorrector();
%block_data.addShadingCorrector(ShadingCorrector(),1:5);

subplot(2,1,2);
imagesc(block_data.loadImage(1));
colormap gray;