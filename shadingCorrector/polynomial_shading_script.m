%POLYNOMIAL SHADING CORRECTION SCRIPT
%Does shading correction using the polynomial surface fitted references
%images. Plots:
    %black smoothed image, black residual
    %grey smoothed image, grey residual
    %white smoothed image, white residual
    %shading corrected image
    %gradient image

clc;
clearvars;
close all;

%instantise an object which loads the data
block_data = BlockData_140316('../data/140316');
%set up shading correction for the data set
block_data.addShadingCorrector(@ShadingCorrector_polynomial,true,[2,2,2]);

reference_name = {'black','white','grey'};

%for each reference image
for i_index = 1:3
    %plot polynomial surface
    fig = figure;
    fig.Position(3:4) = [560,420];
    imagesc(block_data.shading_corrector.reference_image_array(:,:,i_index));
    colormap gray;
    colorbar;
    axis(gca,'off');
    saveas(gca,strcat('reports/figures/shadingCorrection/polynomial_',reference_name{i_index},'.eps'));
    %plot residual
    fig = figure;
    fig.Position(3:4) = [560,420];
    ax = imagesc_truncate(block_data.shading_corrector.getZResidualImage(i_index));
    ax.CLim = [-2,2];
    colorbar;
    axis(gca,'off');
    saveas(gca,strcat('reports/figures/shadingCorrection/residual_',reference_name{i_index},'.png'));
end

%plot the shading corrected sample scan image
fig = figure;
fig.Position(3:4) = [560,420];
imagesc_truncate(block_data.loadSample(1));
colorbar;
colormap gray;
axis(gca,'off');
saveas(gca,'reports/figures/shadingCorrection/polynomial_shadingCorrection.eps');

%plot the gradient in the shading correction
fig = figure;
fig.Position(3:4) = [560,420];
imagesc_truncate(block_data.shading_corrector.b_array);
colorbar;
saveas(gca,'reports/figures/shadingCorrection/polynomial_gradient.png');
axis(gca,'off');