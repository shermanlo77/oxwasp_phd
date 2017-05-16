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
axe_array = cell(1,4);

%number of bins in the frequency density heatmap
n_bin = 100;

%instantise an object which loads the data
block_data = BlockData_140316('data/140316');
%get the pixels which do not belong to the 3d printed sample
threshold = reshape(BlockData_140316.getThreshold_topHalf(),[],1);

%model the mean and variance using gamma glm
model = MeanVar_GLM_identity((block_data.n_sample-1)/2,1);

%array of shading correction names
shading_array = {'no_shad','bw','bgw','polynomial'};

%for each shading correction
for i = 1:4
    %add shading correction to the data
    switch i
        case 2
            block_data.addShadingCorrector(@ShadingCorrector,false);
        case 3
            block_data.addShadingCorrector(@ShadingCorrector,true);
        case 4
            block_data.addShadingCorrector(@ShadingCorrector_polynomial,true,[2,2,2]);
    end
    %if shading correction is applied, interpolate extreme pixels
    if i>1
        block_data.turnOnSetExtremeToNan();
        block_data.turnOnRemoveDeadPixels();
    end
    
    %get the sample mean and variance data
    [sample_mean,sample_var] = block_data.getSampleMeanVar_topHalf();
    %delete data which do not belong to the 3d printed sample
    sample_mean(threshold) = [];
    sample_var(threshold) = [];
    
    %get a range of greyvalues to plot the fit
    x_plot = linspace(min(sample_mean),max(sample_mean),100);
    %train the classifier
    model.train(sample_mean,sample_var);
    %get the variance prediction along with the error bars
    [variance_prediction, up_error, down_error] = model.predict(x_plot');
    
    %plot the frequency density of the mean/var data
    fig = figure;
    fig.InvertHardcopy = 'off'; %export background
    fig.Color = 'white'; %set the background to be white
    ax = plotHistogramHeatmap(sample_mean,sample_var,n_bin);
    hold on;
    %plot the fit/prediction
    plot(x_plot,variance_prediction,'r');
    %plot the error bars
    plot(x_plot,up_error,'r--');
    plot(x_plot,down_error,'r--');
    
    %record the x,y limits of the figure, they will be reused for future figures
    if i == 1
        x_lim = ax.XLim;
        y_lim = ax.YLim;
    else
        %set the x,y limits
        ax.XLim = x_lim;
        ax.YLim = y_lim;

    end
    
    %save the figure to the array
    axe_array{i} = ax;
    
end

%find out the maximum value of the heatmap, let that be c_lim_max
c_lim_max = 0;
%for each figure
for i = 1:4
    %get the c_lim
    c_lim = axe_array{i}.CLim(2);
    %if this is bigger than c_lim_max, updated it
    if c_lim > c_lim_max
        c_lim_max = c_lim;
    end
end
%get the colour of the background
blank_colour = colormap;
blank_colour = blank_colour(1,:);
%set all figures to have the same CLim
for i = 1:4
    axe_array{i}.CLim = [0,c_lim_max];
    axe_array{i}.set('color',blank_colour);
    saveas(axe_array{i},strcat('reports/figures/meanVar/shadingCorrection_',shading_array{i},'.png'),'png');
end