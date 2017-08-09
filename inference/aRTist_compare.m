clc;
clearvars;
close all;

%load data and add shading correction
block_data = AbsBlock_Sep16_120deg();
block_data.addDefaultShadingCorrector();

%get the mean phanton image and aRTist image
phantom = mean(block_data.loadImageStack(),3);
aRTist_uncorrected = block_data.getARTistImage();
aRTist = block_data.getShadingCorrectedARTistImage(ShadingCorrector(),[1,block_data.reference_white-1]);

%get the segmentation image
segmentation = block_data.getSegmentation();
%get the number of segmented images
n_pixel = sum(sum(segmentation));

%plot the phantom and aRTist image
% figure;
% imagesc(phantom);
% colorbar;
% figure;
% imagesc(aRTist_uncorrected);
% colorbar;
% figure;
% imagesc(aRTist);
% colorbar;

%plot aRTist vs phantom greyvalue as a histogram heatmap
% figure;
% hist3Heatmap(reshape(phantom(segmentation),[],1),reshape(aRTist(segmentation),[],1),[300,300],true);
% hold on;
% %get the min and max greyvalue
% min_grey = min([min(min(phantom)),min(min(aRTist))]);
% max_grey = max([max(max(phantom)),max(max(aRTist))]);
% %plot straight line with gradient 1
% plot([min_grey,max_grey],[min_grey,max_grey],'r');
% %label axis
% colorbar;
% xlabel('phantom greyvalue (arb. unit)');
% ylabel('aRTist greyvalue (arb. unit)');

%get random permutation for each image
index = randperm(block_data.n_sample);
%assign half to the training set
%assign other half to the test set
n_train = round(block_data.n_sample/2);
n_test = block_data.n_sample - n_train;
training_index = index(1:n_train);
test_index = index((n_train+1):end);

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
model = MeanVar_GLM_canonical((n_train-1)/2,-2);
model.train(training_mean,training_var);

%declare image, each pixel represent predicted variance, given aRTist
var_predict = zeros(block_data.height, block_data.width);
%for each pixel
for i_column = 1:block_data.width
    for i_row = 1:block_data.height
        %predict the variance and save
        var_predict(i_row,i_column) = model.predict(aRTist(i_row,i_column));
    end
end

%plot the predicted variance
% figure;
% imagesc(var_predict);

%get the test images
test_stack = block_data.loadImageStack(test_index);

for i = 1:n_test
    %for this test image (the 1st one)
    test = test_stack(:,:,i);
    %get the z statistic
    z_image = (test - aRTist)./sqrt(var_predict);
    %plot the z statistics
    % figure;
    % imagesc(z_image);
    % colorbar;

    %work out the p value and plot it
    p_image = 2*(1-normcdf(abs(z_image)));
    %set non segmented pixels to be nan
    p_image(~segmentation) = nan;
    % figure;
    % imagesc(log10(p_image));
    % colorbar;

    %find critical pixels at some level
    critical_index = reshape(significantFDR(reshape(p_image,[],1),normcdf(-5)),block_data.height,block_data.width);
    [critical_y, critical_x] = find(critical_index);

    %plot the phantom scan with critical pixels highlighted
    figure;
    imagesc(test);
    hold on;
    scatter(critical_x, critical_y,'r');
end