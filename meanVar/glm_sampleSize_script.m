%GLM FIT FOR DIFFERENT SAMPLE SIZES
%Fit gamma GLM on the mean and variance relationship. This is done 4
%different times using different sample sizes when calculating the mean and
%variance estimates data. The frequency density is ploted with the fit.

clc;
clearvars;
close all;

%set random seed
rng(uint32(189224219), 'twister');

%instantise an object pointing to the dataset
block_data = BlockData_140316('../data/140316');

%number of bins for the frequency density plot
nbin = 100;

%polynoial order feature
polynomial_order = -1;

%array of sample sizes
n_sample_array = [25,50,75,100];

%array of figures
axe_array = cell(1,numel(n_sample_array));

%for each sample size
for i_sample = 1:numel(n_sample_array)
    
    %get the sample size
    n_sample = n_sample_array(i_sample);
    
    %set n_sample mean/var data
    data_index = randperm(block_data.n_sample);
    data_index = data_index(1:n_sample);
    [sample_mean,sample_var] = block_data.getSampleMeanVar_topHalf(data_index);

    %shape parameter is number of (images - 1)/2, this comes from the chi
    %squared distribution
    shape_parameter = (n_sample-1)/2;

    %model the mean and variance using gamma glm
    model = MeanVar_GLM_canonical(shape_parameter,polynomial_order);
    %train the classifier
    model.train(sample_mean,sample_var,100);

    %get a range of greyvalues to plot the fit
    x_plot = linspace(min(sample_mean),max(sample_mean),100);
    %get the variance prediction along with the error bars
    [variance_prediction, up_error, down_error] = model.predict(x_plot');

    %plot the frequency density
    figure;
    axe_array{i_sample} = plotHistogramHeatmap(sample_mean,sample_var,nbin);
    hold on;
    %plot the fit/prediction
    plot(x_plot,variance_prediction,'r');
    %plot the error bars
    plot(x_plot,up_error,'r--');
    plot(x_plot,down_error,'r--');
    
end

%rescale the colorbar
%declare an array of maximum frequency density, one for each figure or sample size
max_frequency_density = zeros(1,numel(n_sample_array));
%for each sample size
for i_sample = 1:numel(n_sample_array)
    %get the maximum value in the figure
    max_frequency_density(i_sample) = axe_array{i_sample}.CLim(end);
end
%find the maximum maximum value
max_frequency_density = max(max_frequency_density);
%for each figure, set CLim to the that maximum value
for i_sample = 1:numel(n_sample_array)
    axe_array{i_sample}.CLim(end) = max_frequency_density;
end