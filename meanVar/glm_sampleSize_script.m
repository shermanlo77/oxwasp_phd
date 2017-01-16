%GLM FIT FOR DIFFERENT SAMPLE SIZES
%Fit gamma GLM on the mean and variance relationship. This is done 4
%different times using different sample sizes when calculating the mean and
%variance estimates data. The frequency density is ploted with the fit.

clc;
clearvars;
close all;

%number of bins for the frequency density plot
nbin = 100;

%polynoial order feature
polynomial_order = -1;

%array of sample sizes
n_sample_array = [25,50,75,100];

%for each sample size
for i_sample = 1:numel(n_sample_array)
    
    %get the sample size
    n_sample = n_sample_array(i_sample);
    
    %set n_sample mean/var data
    data_index = randperm(100);
    data_index = data_index(1:n_sample);
    [sample_var,sample_mean] = getSampleMeanVar_topHalf('../data/block',data_index);

    %shape parameter is number of (images - 1)/2, this comes from the chi
    %squared distribution
    shape_parameter = (n_sample-1)/2;

    %model the mean and variance using gamma glm
    model = MeanVar_GLM_canonical(shape_parameter,polynomial_order);
    %train the classifier
    model.train(sample_var,sample_mean,100);

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
