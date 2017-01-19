%FIT GLM SCRIPT
%Model the mean and variance relationship, using the top half of the stack image The response variance is gamma
%distributed with known shape parameter from the chi squared distribution.
%The feature is the mean grey value to some power. A frequency
%density graph is plotted along with the fit

clc;
clearvars;
close all;

%number of bins for the frequency density plot
nbin = 100;

%instantise an object pointing to the dataset
block_data = BlockData_140316('../data/140316');

%get variance mean data of the top half of the scans (images 1 to 100)
[sample_mean,sample_var] = block_data.getSampleMeanVar_topHalf();

%shape parameter is number of (images - 1)/2, this comes from the chi
%squared distribution
shape_parameter = (block_data.n_sample-1)/2;

%for each polynomial order
for polynomial_order = [-4,-3,-2,-1]
    
    %model the mean and variance using gamma glm
    model = MeanVar_GLM_canonical(shape_parameter,polynomial_order);
    %train the classifier
    model.train(sample_mean,sample_var,100);

    %get a range of greyvalues to plot the fit
    x_plot = linspace(min(sample_mean),max(sample_mean),100);
    %get the variance prediction along with the error bars
    [variance_prediction, up_error, down_error] = model.predict(x_plot');

    %plot the frequency density
    plotHistogramHeatmap(sample_mean,sample_var,nbin);
    hold on;
    %plot the fit/prediction
    plot(x_plot,variance_prediction,'r');
    %plot the error bars
    plot(x_plot,up_error,'r--');
    plot(x_plot,down_error,'r--');
end