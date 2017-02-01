%TRAINING TEST GLM SCRIPT
%Fit gamma glm on the mean variance data

clc;
clearvars;
close all;

%set random seed
rng(uint32(33579150), 'twister');

%instantise an object pointing to the dataset
block_data = BlockData_140316('../data/140316');
%segment the mean variance data to only include the 3d printed sample,
%threshold indicate pixels which belong to the background
threshold = reshape(BlockData_140316.getThreshold_topHalf(),[],1);

%array of polynomial features to explore
polynomial_array = [-4,-3,-2,-1];

%number of times to spilt the training/test set
n_repeat = 20;

%array to store the mse for each polynomial and each spilt
    %dim1: for each repeat
    %dim2: for each polynomial order
mse_training_array = zeros(n_repeat,numel(polynomial_array)); %training mse
mse_test_array = zeros(n_repeat,numel(polynomial_array)); %test mse

%shape parameter is number of (images - 1)/2, this comes from the chi
%squared distribution
shape_parameter = (50-1)/2;

%for each polynomialk order
for i_polynomial = 1:numel(polynomial_array)
    
    %get the polynomial order
    polynomial_order = polynomial_array(i_polynomial);
    
    %model the mean and variance using gamma glm
    model = MeanVar_GLM_canonical(shape_parameter,polynomial_order);
    
    %for n_repeat times
    for i_repeat = 1:n_repeat

        %get random index of the training and test data
        index_suffle = randperm(100);
        training_index = index_suffle(1:50);
        test_index = index_suffle(51:100);
        
        %get variance mean data of the training set
        [sample_mean,sample_var] = block_data.getSampleMeanVar_topHalf(training_index);
        %segment the mean var data
        sample_mean(threshold) = [];
        sample_var(threshold) = [];
        
        %train the classifier
        model.train(sample_mean,sample_var,100);
        %get the training mse
        mse_training_array(i_repeat,i_polynomial) = model.getPredictionMSE(sample_mean,sample_var);

        %get the variance mean data of the test set
        [sample_mean,sample_var] = block_data.getSampleMeanVar_topHalf(test_index);
        %segment the mean var data
        sample_mean(threshold) = [];
        sample_var(threshold) = [];
        %get the test mse
        mse_test_array(i_repeat,i_polynomial) = model.getPredictionMSE(sample_mean,sample_var);
        
    end
    
end

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