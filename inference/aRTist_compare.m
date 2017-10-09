clc;
clearvars;
close all;

%set random seed
rng(uint32(3538096789), 'twister');

%load data and add shading correction
block_data = AbsBlock_Sep16_120deg();
block_data.addDefaultShadingCorrector();

%get random permutation for each image
index = randperm(block_data.n_sample);
n_train = block_data.n_sample - 2;
n_test = 1;
n_calibrate = 1;
training_index = index(1:n_train);
test_index = index((n_train+1):(n_train+n_test));
calibrate_index = index((n_train+n_test+1):end);

%get a phanton image and aRTist image
phantom = mean(block_data.loadImageStack(calibrate_index),3);
aRTist = block_data.getShadingCorrectedARTistImage(ShadingCorrector(),1:block_data.reference_white);

%get the segmentation image
segmentation = block_data.getSegmentation();
%get the number of segmented images
n_pixel = sum(sum(segmentation));

phantom_vector = reshape(phantom(segmentation),[],1);
aRTist_vector = reshape(aRTist(segmentation),[],1);
d = phantom_vector - aRTist_vector;

%plot aRTist vs phantom greyvalue as a histogram heatmap
figure;
hist3Heatmap(phantom_vector,aRTist_vector,[300,300],true);
hold on;
%get the min and max greyvalue
min_grey = min([min(min(phantom)),min(min(aRTist))]);
max_grey = max([max(max(phantom)),max(max(aRTist))]);
%plot straight line with gradient 1
plot([min_grey,max_grey],[min_grey,max_grey],'r');
%label axis
colorbar;
xlabel('phantom greyvalue (arb. unit)');
ylabel('aRTist greyvalue (arb. unit)');

%plot phantom - aRTist vs aRTist greyvalue as a histogram heatmap
figure;
hist3Heatmap(aRTist_vector,d,[100,100],true);
hold on;
%get the min and max greyvalue
min_grey = min([min(min(phantom)),min(min(aRTist))]);
max_grey = max([max(max(phantom)),max(max(aRTist))]);
%plot straight line with gradient 1
plot([min_grey,max_grey],[min_grey,max_grey],'r');
%label axis
colorbar;
xlabel('aRTist greyvalue (arb. unit)');
ylabel('difference in greyvalue (arb. unit)');

model = MeanVar_kNN(1E3);
model.train(aRTist_vector,d);
aRTist_plot = (min(aRTist_vector):max(aRTist_vector))';
hold on;
plot(aRTist_plot,model.predict(aRTist_plot),'-r');

d_predict = model.predict(reshape(aRTist,[],1));
aRTist = aRTist + reshape(d_predict,block_data.height,block_data.width);

aRTist_vector = reshape(aRTist(segmentation),[],1);
d = phantom_vector - aRTist_vector;

%plot the phantom and aRTist image
figure;
imagesc(phantom);
colorbar;
figure;
imagesc(aRTist);
colorbar;

%plot aRTist vs phantom greyvalue as a histogram heatmap
figure;
hist3Heatmap(phantom_vector,aRTist_vector,[300,300],true);
hold on;
%get the min and max greyvalue
min_grey = min([min(min(phantom)),min(min(aRTist))]);
max_grey = max([max(max(phantom)),max(max(aRTist))]);
%plot straight line with gradient 1
plot([min_grey,max_grey],[min_grey,max_grey],'r');
%label axis
colorbar;
xlabel('phantom greyvalue (arb. unit)');
ylabel('aRTist greyvalue (arb. unit)');

%get the training images
training_stack = block_data.loadImageStack(training_index);
%segment the image
training_stack = reshape(training_stack,block_data.area,n_train);
training_stack = training_stack(reshape(segmentation,[],1),:);
%get the segmented mean and variance greyvalue
training_mean = mean(training_stack,2);
training_var = var(training_stack,[],2);
%plot the variance vs mean
% figure;
% hist3Heatmap(training_mean,training_var,[100,100],true);

%train glm using the training set mean and variance
model = MeanVar_GLM((n_train-1)/2,1,LinkFunction_Identity());
model.train(training_mean,training_var);

%predict variance given aRTist
var_predict = reshape(model.predict(reshape(aRTist,[],1)),block_data.height, block_data.width);

%plot the predicted variance
% figure;
% imagesc(var_predict);

%get the test images
test_stack = block_data.loadImageStack(test_index);

for i = 1:n_test
    %for this test image (the 1st one)
    test = test_stack(:,:,i);
    var_predict = reshape(model.predict(reshape(test,[],1)),block_data.height, block_data.width);
    %get the z statistic
    z_image = (test - aRTist)./sqrt(var_predict);
    
    %set non segmented pixels to be nan
    z_image(~segmentation) = nan;
    
    %work out the p value and plot it
    p_image = 2*(1-normcdf(abs(z_image)));
    
    figure;
    imagesc(p_image);
    colorbar;
    
    m = sum(sum(~isnan(z_image)));
    
    %histogram
    z_vector = reshape(z_image,[],1);
    z_vector(isnan(z_vector)) = [];
    z_plot = linspace(min(z_vector),max(z_vector),1000);
    figure;
    histogram(z_vector,'Normalization','CountDensity');
    hold on;
    plot(z_plot,normpdf(z_plot)*m);
    xlabel('z statistic');
    ylabel('frequency density');
    
    %qqplot
    figure;
    scatter(norminv(((1:m)-0.5)/m),sort(z_vector),'x');
    hold on;
    plot([min(z_vector),max(z_vector)],[min(z_vector),max(z_vector)],'r--');
    xlabel('Standard Normal quantiles');
    ylabel('z statistics quantiles');

    %find critical pixels at some level
    critical_index = reshape(significantFDR(reshape(p_image,[],1),normcdf(-2),true),block_data.height,block_data.width);
    [critical_y, critical_x] = find(critical_index);

    %plot the phantom scan with critical pixels highlighted
    figure;
    imagesc(test);
    hold on;
    scatter(critical_x, critical_y,'r.');
    colorbar;
end