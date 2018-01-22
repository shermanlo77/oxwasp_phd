clc;
clearvars;
close all;

%set random seed
rng(uint32(139952197), 'twister');

%get the data
block_data = AbsBlock_Mar16();
%get the threshold of the top half of the image, threshold of the 3d printed sample
threshold = AbsBlock_Mar16.getThreshold_topHalf();

%get the number of images in the entire dataset
n = block_data.n_sample;

n_repeat = 4;

%declare array for storing the z statistics and p values for each
z_array = zeros(block_data.height, block_data.width, n_repeat*25);

%instantiate GLm model
model = MeanVar_GLM_identity((25-1)/2,1);

for i_repeat = 1:n_repeat
    
    %shuffle the order of the data
    index = randperm(n);

    %assign 50 images to the model set
    model_index = index(1:50);
    %assign 25 images to represent the ground truth
    true_index = index(51:75);
    %assign 25 images to the test set
    test_index = index(76:end);
    
    %get variance mean data of the training set
    [sample_mean,sample_var] = block_data.getSampleMeanVar_topHalf(model_index);
    %segment the mean var data
    segmentation = block_data.getSegmentation();
    segmentation = segmentation(1:(block_data.height/2),:);
    segmentation = reshape(segmentation,[],1);
    sample_mean = sample_mean(segmentation);
    sample_var = sample_var(segmentation);

    %train the classifier to predict the variance
    model.train(sample_mean,sample_var);
    
    %take the mean over the 25 images in the ground truth set
    true_image = mean(block_data.loadImageStack(true_index),3);
    
    
    %for each test image
    for i_test = 1:numel(test_index)

        %get the test image
        test_image = block_data.loadImage(test_index(i_test));

        %get the std prediction given the test image
        error = sqrt(reshape(model.predict(reshape(test_image,[],1)),block_data.height,block_data.width));

        %calculate the z statistic
        z = (test_image - true_image) ./ error; %something weird when . removed

        %save the z statistic to the array
        z_array(:, :, (i_repeat-1)*numel(test_index) + i_test) = z;
    end
    
end


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
        p_image(i_height, i_width) = chi2gof_norm(z_vector, 10, false);
    end
end

%plot the p values from the chi squared goodness of fit test
fig = figure;
imagesc(p_image);
colorbar;
%get the critical value at the 2 sigma significance level
p_critical = normcdf(-2) / block_data.area;
%find significant pixels
[y,x] = find(p_image < p_critical);
hold on;
%scatter plot significant pixels
scatter(x,y,'r','filled');
axis(gca,'off');
saveas(fig,'reports/figures/meanVar/chi_squared_critical.png','png');

%plot histogram of p values
fig = figure;
histogram(reshape(p_image, [], 1), 20, 'Normalization', 'countdensity');
xlabel('p values');
ylabel('Frequency density');
saveas(fig,'reports/figures/meanVar/chi_squared_p_values.eps','epsc');