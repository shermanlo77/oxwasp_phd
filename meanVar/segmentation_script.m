%SEGMENTATION SCRIPT
%The 3d printed sample is segmentated using a threshold. The procedure is
%as follows:
    %do shading correction using median filter on the reference images
    %take the mean over all shading corrected images
    %remove dead pixels
    %remove pixels with greyvalues more than 4.7E4

clc;
clearvars;
close all;

%load data and shading correction
block_data = BlockData_140316('../data/140316');
block_data.addShadingCorrector(@ShadingCorrector_median,1,[3,3,3]);

%load the images
slice = block_data.loadSampleStack();
%crop the images to keep the top half
slice = slice(1:(round(block_data.height/2)),:,:);
%take the mean over all images
slice = mean(slice,3);
%remove dead pixels
slice = removeDeadPixels(slice);

%indicate pixels with greyvalues more than 4.7E4
threshold = slice>4.7E4;
%set these pixels to be nan
slice(threshold) = nan;
%plot the threshold image
figure;
imagesc(slice);

%reshape the threshold indicator to be a vector
threshold = reshape(threshold,[],1);

%turn shading correction off to get the sample mean and variance data
block_data.turnOffShadingCorrection();
[sample_mean, sample_var] = block_data.getSampleMeanVar_topHalf();

%delete thresholded pixels
sample_mean(threshold) = [];
sample_var(threshold) = [];

%plot the sample mean and sample variance as a heatmap
figure;
plotHistogramHeatmap(sample_mean,sample_var,100);