%TRAINING TEST GLM SCRIPT (on data post shading corrected)
%Fit gamma glm on the mean variance training data and then predict the
%variance given the mean on the test set. The training and test MSE is
%presented for a number of different polynomial features.

clc;
clearvars;
close all;

%set random seed
rng(uint32(267632689), 'twister');

%instantise an object pointing to the dataset
block_data = BlockData_140316('../data/140316');
%turn on shading correction
block_data.addShadingCorrector(@ShadingCorrector,true);

%number of images use to get the training set mean and variance
n_train = 50;

%number of times to spilt the training/test set
n_repeat = 20;

%array of polynomial features to explore
polynomial_array = [-4,-3,-2,-1];

%create an array of glm objects, each with different polynomial features
glm_array = cell(1,numel(polynomial_array));
%define the shape parameter of the gamma glm
shape_parameter = (n_train-1)/2;
%for each polynomial feature
for i_polynomial = 1:numel(polynomial_array)
    %instantise a glm object with that polynomial feature
    glm_array{i_polynomial} = MeanVar_GLM_canonical(shape_parameter,polynomial_array(i_polynomial));
end

%do training/test mse on predicting the variance given the mean
[mse_training_array, mse_test_array] = trainingTestMeanVar(block_data, glm_array, n_train, n_repeat);

%box plot the training mse
fig = figure;
boxplot(mse_training_array,polynomial_array);
ax_train = fig.CurrentAxes;
xlabel(ax_train,'Polynomial order');
ylabel(ax_train,'Training MSE');

%box plot the test mse
fig = figure;
boxplot(mse_test_array,polynomial_array);
ax_test = fig.CurrentAxes;
xlabel(ax_test,'Polynomial order');
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