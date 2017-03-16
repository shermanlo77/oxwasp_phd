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
%get the threshold logic image, this is a matrix of logics which indicate
%pixels which are from the background
threshold = BlockData_140316.getThreshold_topHalf();

%load the image
slice = block_data.loadSampleStack();
%crop the image, retaining the top half
slice = slice(1:(block_data.height/2),:,:);
%take the mean over all images
slice = mean(slice,3);
%set background pixels to be nan
slice(threshold) = nan;
%plot the threshold image
fig = figure;
ax = imagesc(slice);
colormap gray;
colorbar;
axis(gca,'off');
saveas(fig,'reports/figures/meanVar/segment.eps');

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
