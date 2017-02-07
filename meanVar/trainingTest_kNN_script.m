%TRAINING TEST kNN SCRIPT
%Do regression on the mean and variance data using kNN neighbours. The
%training and test MSE from prediction is presented for different ks. The
%mean and variance were obtained from taking the sample mean and sample
%variance over all images in a set. The images were spilt into 50:50
%training/test set

clc;
clearvars;
close all;

%set random seed
rng(uint32(235567122), 'twister');

%instantise an object pointing to the dataset
block_data = BlockData_140316('../data/140316');

%array of k to explore
k_array = [1E2,1E3,1E4,1E5];

%number of images use to get the training set mean and variance
n_train = 50;

%number of times to spilt the training/test set
n_repeat = 20;

%do training/test mse on predicting the variance given the mean
[mse_training_array, mse_test_array] = trainingTestMeanVar(block_data, @MeanVar_kNN, n_train, k_array, n_repeat);

%box plot the training mse
fig = figure;
boxplot(mse_training_array,k_array);
ax_train = fig.CurrentAxes;
xlabel(ax_train,'k');
ylabel(ax_train,'Training MSE');

%box plot the test mse
fig = figure;
boxplot(mse_test_array,k_array);
ax_test = fig.CurrentAxes;
xlabel(ax_test,'k');
ylabel(ax_test,'Test MSE');

%rescale the y axis
%ylim_array is an array of ylim of each boxplot
ylim_array = zeros(2,2);
%for each box plot, get the ylim and save it
ylim_array(1,:) = ax_train.YLim;
ylim_array(2,:) = ax_test.YLim;
%ylim covers the whole range of y of the two boxplots
ylim = zeros(1,2);
ylim(1) = min(ylim_array(:,1));
ylim(2) = max(ylim_array(:,2));
%rescale the y axis
ax_train.YLim = ylim;
ax_test.YLim = ylim;