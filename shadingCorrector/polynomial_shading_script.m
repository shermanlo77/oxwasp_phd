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
block_data.addShadingCorrector(@ShadingCorrector_polynomial,[2,2,2]);

%for each reference image
for i_index = 1:3
    %plot polynomial surface
    figure;
    imagesc(block_data.shading_corrector.reference_image_array(:,:,i_index));
    colormap gray;
    %plot residual
    ax = imagesc_truncate(block_data.shading_corrector.getZResidualImage(i_index));
    ax.CurrentAxes.CLim = [-2,2];
    colorbar;
end

%plot the shading corrected sample scan image
imagesc_truncate(block_data.loadSample(1));
colorbar;
colormap gray;

%plot the gradient in the shading correction
imagesc_truncate(block_data.shading_corrector.b_array);
colorbar;