%MEAN VARIANCE RELATIONSHIP BEFORE/AFTER SHADING CORRECTION SCRIPT
%Plots the sample mean and sample variance frequency density heat map
%before and after shading correction. A number of shading correction were
%considered:
    %unsmoothed b/w
    %unsmoothed b/g/w
    %polynomial smoothed order 2/2/2 on b/g/w

clc;
clearvars;
close all;

%declare array of figures
figure_array = cell(1,4);

%number of bins in the frequency density heatmap
n_bin = 100;

%instantise an object which loads the data
block_data = BlockData_140316('../data/140316');

%UNSHADING CORRECTED
%get the sample mean and sample variance data
[sample_mean,sample_var] = block_data.getSampleMeanVar_topHalf();
%plot the frequency density of the mean/var data
ax = plotHistogramHeatmap(sample_mean,sample_var,n_bin);
%record the x,y limits of the figure, they will be reused for future figures
x_lim = ax.CurrentAxes.XLim;
y_lim = ax.CurrentAxes.YLim;
%save the figure to the array
figure_array{1} = ax;

%SHADING CORRECTED
%UNSMOOTHED B/W
%instantise shading corrector (consider grey image = false)
block_data.addShadingCorrector(@ShadingCorrector,false);
%get the sample_mean and sample_var data, plot frequency density heatmap
[sample_mean,sample_var] = block_data.getSampleMeanVar_topHalf();
ax = plotHistogramHeatmap(sample_mean,sample_var,n_bin,x_lim,y_lim);
%set the x,y limits
ax.CurrentAxes.XLim = x_lim;
ax.CurrentAxes.YLim = y_lim;
%save the figure to the array
figure_array{2} = ax;

%SHADING CORRECTED
%UNSMOOTHED B/G/W
%instantise shading corrector (consider grey image = true)
block_data.addShadingCorrector(@ShadingCorrector,true);
%get the sample_mean and sample_var data, plot frequency density heatmap
[sample_mean,sample_var] = block_data.getSampleMeanVar_topHalf();
ax = plotHistogramHeatmap(sample_mean,sample_var,n_bin,x_lim,y_lim);
%set the x,y limits
ax.CurrentAxes.XLim = x_lim;
ax.CurrentAxes.YLim = y_lim;
%save the figure to the array
figure_array{3} = ax;

%SHADING CORRECTED
%POLYNOMIAL SMOOTHED 2/2/2 B/G/W
%instantise shading corrector (consider grey image = true) (polynomial order = 2)
block_data.addShadingCorrector(@ShadingCorrector_polynomial,true,[2,2,2]);
%get the sample_mean and sample_var data, plot frequency density heatmap
[sample_mean,sample_var] = block_data.getSampleMeanVar_topHalf();
ax = plotHistogramHeatmap(sample_mean,sample_var,n_bin,x_lim,y_lim);
%set the x,y limits
ax.CurrentAxes.XLim = x_lim;
ax.CurrentAxes.YLim = y_lim;
%save the figure to the array
figure_array{4} = ax;

%find out the maximum value of the heatmap, let that be c_lim_max
c_lim_max = 0;
%for each figure
for i = 1:4
    %get the c_lim
    c_lim = figure_array{i}.CurrentAxes.CLim(2);
    %if this is bigger than c_lim_max, updated it
    if c_lim > c_lim_max
        c_lim_max = c_lim;
    end
end
%set all figures to have the same CLim
for i = 1:4
    figure_array{i}.CurrentAxes.CLim = [0,c_lim_max];
end

%it can be observed there are two modes in the background, plot the
%shading corrected and uncorrected image with a change of scale to
%highlight one of the modes
block_data.turnOffShadingCorrection();
for i = 1:2
    figure;
    imagesc(block_data.loadSample(1),[4.8E4,5.1E4]);
    colorbar;
    colormap gray;
    block_data.turnOnShadingCorrection();
end