%BLOCK SHADING CORRECTION SCRIPT
%Does shading correction using the given algorithm, using the mean white, mean gray and mean black image.
%Plots the scan and shading corrected scan.
%Also plots the gradient of target greyscale vs reference greyscale in shading correction.

clc;
clearvars;
close all;

%instantise an object which loads the data
block_data = BlockData_140316();

%plot the sample scan image
figure;
ax_originial = imagesc_truncate(block_data.loadSample(1));
colorbar;
colormap gray;
axis(gca,'off');
saveas(gca,'reports/figures/shadingCorrection/block.png');

%add shading correction for the data set
block_data.addShadingCorrector(ShadingCorrector());

%plot the shading corrected sample scan image
figure;
ax = imagesc_truncate(block_data.loadSample(1));
ax.CLim = ax_originial.CLim;
colorbar;
colormap gray;
axis(gca,'off');
saveas(gca,'reports/figures/shadingCorrection/block_shadingCorrected.png');

%get the gradient in the shading correction
grad = block_data.shading_corrector.b_array;
%plot the gradient in the shading correction
figure;
imagesc_truncate(grad);
colorbar;
hold on;
%get the coordinates of nan and scatter plot it
[y_nan, x_nan] = find(isnan(grad));
scatter(x_nan, y_nan, 'r');
%get the coordinates of negative gradient, and scatter plot it
[y_nve, x_nve] = find(grad<0);
scatter(x_nve, y_nve, 'g');
axis(gca,'off');
saveas(gca,'reports/figures/shadingCorrection/gradient.png');