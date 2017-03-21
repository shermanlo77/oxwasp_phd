clc;
clearvars;
close all;

%Z STATISTICS SCRIPT
%Spilt the data into 3 parts
    %Part 1: fit mean var GLM model to the data, using top half of the image
    %Part 2: 25 images, take the mean of it and treat it as the ground truth
    %Part 3: use the model to predict the variance, for each of the 25 images, calculate the z statistic
%Plots:
    %z statistics for each of the 25 test images
    %significant pixels for each of the 25 test images at the 5 sigma level
    %p values from the chi squared goodness of fit test, circled are significant pixels at the 2 sigma level

%set random seed
rng(uint32(5949338), 'twister');

%get the data
block_data = BlockData_140316('../data/140316');
%add bgw shading correction
block_data.addShadingCorrector(@ShadingCorrector,true);
block_data.turnOnRemoveDeadPixels();
block_data.turnOnSetExtremeToNan();

%get the threshold of the top half of the image, threshold of the 3d printed sample
threshold = BlockData_140316.getThreshold_topHalf();

%get the number of images in the entire dataset
n = block_data.n_sample;

%shuffle the order of the data
index = randperm(n);

%assign 50 images to the model set
model_index = index(1:50);
%assign 25 images to represent the ground truth
true_index = index(51:75);
%assign 25 images to the test set
test_index = index(76:end);

%declare array for storing the z statistics and p values for each of the 25 test images
z_array = zeros(block_data.height, block_data.width, numel(test_index));
p_array = zeros(block_data.height, block_data.width, numel(test_index));

%instantiate GLm model
model = MeanVar_GLM_identity((numel(model_index)-1)/2,1);

%get variance mean data of the training set
[sample_mean,sample_var] = block_data.getSampleMeanVar_topHalf(model_index);
%segment the mean var data
sample_mean(threshold) = [];
sample_var(threshold) = [];

%train the classifier to predict the variance
model.train(sample_mean,sample_var);

%take the mean over the 25 images in the ground truth set
true_image = mean(block_data.loadSampleStack(true_index),3);

%for each test image
for i_test = 1:numel(test_index)

    %get the test image
    test_image = block_data.loadSample(test_index(i_test));

    %get the std prediction given the test image
    error = sqrt(reshape(model.predict(reshape(test_image,[],1)),block_data.height,block_data.width));

    %calculate the z statistic
    z = (test_image - true_image) ./ error; %something weird when . removed
    
    %save the z statistic to the array
    z_array(:,:,i_test) = z;
    %save the p value to the array
    p_array(:,:,i_test) = normcdf(z);
end

%for 6 test image, plot the p value
fig = figure;
for i = 1:6
    subplot(3,2,i);
    imagesc(p_array(:,:,i),[0,1]);
    colorbar;
    axis(gca,'off');
end
saveas(fig,'reports/figures/meanVar/p_values.eps','epsc');

%define the significance level
sig_level = normcdf(-5) / block_data.area;
%define the critical p value
p_critical = [sig_level/2, 1-sig_level/2];

%for 6 test image, plot the scan with circled significant pixels
fig = figure;
fig.Position(3:4) = [395, 450];
for i = 1:6
    %plot the scan
    subplot(3,2,i,imagesc_truncate(block_data.loadSample(test_index(i))));
    axis(gca,'off');
    colormap gray;
    hold on;
    %get the p values
    p_image = p_array(:,:,i);
    %find p values which are significant
    p_significant = (p_image < p_critical(1)) | (p_critical(2) < p_image);
    [y,x] = find(p_significant);
    %plot the significant pixels as a scatter plot
    scatter(x,y,'r');
end
saveas(fig,'reports/figures/meanVar/critical_values.eps','epsc');

%---CHI SQUARED GOODNESS OF FIT TEST---%

%p_image is an image containing the p values of each pixel from the chi
%squared goodness of fit test
p_image = zeros(block_data.height, block_data.width);
%for each pixel in the image
for i_width = 1:block_data.width
    for i_height = 1:block_data.height
        %get the z statistic from each test image
        z_vector = reshape(z_array(i_height, i_width, :), [], 1);
        %do the chi squared goodness of fit test, save the p value in p_image
        p_image(i_height, i_width) = chi2gof_norm(z_vector, 5, false);
    end
end

%plot the p values from the chi squared goodness of fit test
figure;
imagesc(p_image);
colorbar;
%get the critical value at the 2 sigma significance level
p_critical = normcdf(-2) / block_data.area;
%find significant pixels
[y,x] = find(p_image < p_critical);
hold on;
%scatter plot significant pixels
scatter(x,y,'r','filled');

%plot histogram of p values
figure;
histogram(reshape(p_image, [], 1), 20, 'Normalization', 'countdensity');
xlabel('p values');
ylabel('Frequency density');