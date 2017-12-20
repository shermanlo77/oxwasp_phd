clc;
clearvars;
close all;

%set random seed
rng(uint32(335747226), 'twister');

%load data and add shading correction
block_data = AbsBlock_July16_30deg();
block_data.addDefaultShadingCorrector();

%get random permutation for each image
index = randperm(block_data.n_sample);
n_train = round(block_data.n_sample/2);
n_artist = block_data.n_sample - n_train;
%get the segmentation image
segmentation = block_data.getSegmentation();
%get the number of segmented images
n_pixel = sum(sum(segmentation));

meanvar_index = index(1:(n_train-1));
test_index = index(n_train-1);
artist_index = index((n_train+1):end);

aRTist = mean(block_data.loadImageStack(artist_index),3);
[x_grid, y_grid] = meshgrid(1:block_data.width, 1:block_data.height);
plane = (1E4/(sqrt(2)*1000)) * (x_grid - block_data.width/2) + (1E4/(sqrt(2)*1000)) * (y_grid - block_data.height/2);

figure;
imagesc(aRTist);
colorbar;

%get the training images
training_stack = block_data.loadImageStack(meanvar_index);
%segment the image
training_stack = reshape(training_stack,block_data.area,n_train-1);
training_stack = training_stack(reshape(segmentation,[],1),:);
%get the segmented mean and variance greyvalue
training_mean = mean(training_stack,2);
training_var = var(training_stack,[],2);
% %plot the variance vs mean
% figure;
% hist3Heatmap(training_mean,training_var,[100,100],true);

%train glm using the training set mean and variance
model = MeanVar_GLM((n_train-2)/2,1,LinkFunction_Identity());
model.train(training_mean,training_var);

%predict variance given aRTist
var_predict = reshape(model.predict(reshape(aRTist,[],1)),block_data.height, block_data.width);

%get the test images
test = block_data.loadImageStack(test_index);
test = test + plane;
defect_value = 4E3;
test(962:1038, 962:1038) = test(962:1038, 962:1038) + defect_value;
defect_noise = model.predict(mean(reshape(test(962:1038, 962:1038),[],1)));
test(962:1038, 962:1038) = test(962:1038, 962:1038) + normrnd(0,sqrt(defect_noise),77,77);

z_image = (test - aRTist)./sqrt(var_predict);
z_image(~segmentation) = nan;

convolution = EmpericalConvolution(z_image,20, 20, [200,200]);
convolution.estimateNull(1000);
convolution.setMask(segmentation);
convolution.doTest();

fig = figure;
imagesc(-log10(convolution.p_image));
colorbar;
hold on;
colorbar;
fig.CurrentAxes.XTick = [];
fig.CurrentAxes.YTick = [];

fig = figure;
imagesc(test);
colorbar;
hold on;
colorbar;
fig.CurrentAxes.XTick = [];
fig.CurrentAxes.YTick = [];

fig = figure;
imagesc(test);
colorbar;
hold on;
colorbar;
[critical_y, critical_x] = find(convolution.sig_image);
scatter(critical_x, critical_y,'r.');
fig.CurrentAxes.XTick = [];
fig.CurrentAxes.YTick = [];

fig = figure;
imagesc(convolution.mean_null);
colorbar;
fig.CurrentAxes.XTick = [];
fig.CurrentAxes.YTick = [];

fig = figure;
imagesc(sqrt(convolution.var_null));
colorbar;
fig.CurrentAxes.XTick = [];
fig.CurrentAxes.YTick = [];