%FIT kNN SCRIPT
%Regress the mean and variance relationship using k nearest neighbours for
%different k. The frequency density is plotted with the fit

clc;
clearvars;
close all;

%number of bins for the frequency density plot
nbin = 100;

%instantise an object pointing to the dataset
block_data = BlockData_140316('data/140316');

%get variance mean data of the top half of the scans (images 1 to 100)
[sample_mean,sample_var] = block_data.getSampleMeanVar_topHalf();

%segment the mean variance data to only include the 3d printed sample
threshold = reshape(BlockData_140316.getThreshold_topHalf(),[],1);
sample_mean(threshold) = [];
sample_var(threshold) = [];

%for each k
for k = [1E2,1E3,1E4,1E5]
    
    %model the mean and variance using kNN
    model = MeanVar_kNN(k);
    %train the classifier
    model.train(sample_mean,sample_var)

    %get a range of greyvalues to plot the fit
    x_plot = model.mean_lookup_start + (0:model.n_lookup);
    %get the variance prediction
    variance_prediction = model.variance_lookup;

    %plot the frequency density
    figure;
    plotHistogramHeatmap(sample_mean,sample_var,nbin);
    hold on;
    %plot the fit/prediction
    plot(x_plot,variance_prediction,'r');
end