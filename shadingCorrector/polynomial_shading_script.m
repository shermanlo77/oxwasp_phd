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

%declare an array reference_stack which stores the mean black and mean
%white image
reference_stack = zeros(block_data.height,block_data.width,2);
%load and save the mean black image
reference_stack(:,:,1) = mean(block_data.loadBlackStack(),3);
%load and save the mean grey image
reference_stack(:,:,2) = mean(block_data.loadGreyStack(),3);
%load and save the mean white image
reference_stack(:,:,3) = mean(block_data.loadWhiteStack(),3);

%instantise a shading corrector and set it up using thethis.n_panel_column reference images
shading_corrector = ShadingCorrector_polynomial(reference_stack);

%start counting the panels
block_data.resetPanelCorner();
%for each panel
while block_data.hasNextPanelCorner()
    %get the coordinates of the panel
    panel_corners = block_data.getNextPanelCorner();
    %for each reference image, fit polynomial onto that panel
    for i_index = 1:3
        shading_corrector.smoothPanel(i_index,panel_corners,2);
    end
end

%for each reference image
for i_index = 1:3
    %plot polynomial surface
    figure;
    imagesc(shading_corrector.reference_image_array(:,:,i_index));
    colormap gray;
    %plot residual
    ax = imagesc_truncate(shading_corrector.getZResidualImage(i_index));
    ax.CurrentAxes.CLim = [-2,2];
    colorbar;
end

%set the shading corrector to do shading correction
shading_corrector.calibrate();

%get the first scan image of the sample
sample_scan = double(block_data.loadSample(1));

%plot the shading corrected sample scan image
imagesc_truncate(shading_corrector.shadeCorrect(sample_scan));
colorbar;
colormap gray;

%plot the gradient in the shading correction
imagesc_truncate(shading_corrector.b_array);
colorbar;