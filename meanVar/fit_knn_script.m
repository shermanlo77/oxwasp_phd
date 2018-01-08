%FIT kNN SCRIPT
%Regress the mean and variance relationship using k nearest neighbours for
%different k. The frequency density is plotted with the fit

clc;
clearvars;
close all;

%number of bins for the frequency density plot
nbin = 100;

%instantise an object pointing to the dataset
block_data = AbsBlock_Mar16();

%get variance mean data of the top half of the scans (images 1 to 100)
[sample_mean,sample_var] = block_data.getSampleMeanVar_topHalf();

%segment the mean variance data to only include the 3d printed sample
segmentation = block_data.getSegmentation();
segmentation = segmentation(1:(block_data.height/2),:);
segmentation = reshape(segmentation,[],1);
sample_mean = sample_mean(segmentation);
sample_var = sample_var(segmentation);

x_plot = linspace(min(sample_mean),max(sample_mean),500);

%for each k
for k = [1E1, 1E3]
    
    %model the mean and variance using kNN
    model = KernelRegression(EpanechnikovKernel(), k);
    %train the classifier
    model.train(sample_mean,sample_var);

    %get the variance prediction
    variance_prediction = model.predict(x_plot);

    %plot the frequency density
    figure;
    hist3Heatmap(sample_mean,sample_var,[nbin,nbin],false)
    hold on;
    %plot the fit/prediction
    plot(x_plot,variance_prediction,'r');
end