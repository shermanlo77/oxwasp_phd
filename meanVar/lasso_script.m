clc;
clearvars;
close all;

block_data = AbsBlock_Sep16_120deg();
n_image = block_data.n_sample;
n_train = round(n_image / 2);
n_test = n_image - n_train;

index = randperm(n_image);
training_index = index(1:n_train);
test_index = index((n_train+1):end);

%get the segmentation
segmentation = block_data.getSegmentation();
%get the number of segmented pixels
n_pixel = sum(sum(segmentation));

training_stack = block_data.loadImageStack(training_index);
training_stack = reshape(training_stack,block_data.area,n_train);
training_stack = training_stack(segmentation,:);

mean_train = mean(training_stack,2);
var_train = var(training_stack,[],2);

polynomial_orders = [-2,-1,1,2];
X = zeros(n_pixel,numel(polynomial_orders));
for i = 1:numel(polynomial_orders)
    X(:,i) = mean_train.^(polynomial_orders(i));
end
mean_feature = mean(X,1);
std_feature = std(X,[],1);
X = (X - repmat(mean_feature,n_pixel,1)) ./ repmat(std_feature,n_pixel,1);

%[B,FitInfo] = lassoglm(X,var_train);

figure;
hist3Heatmap(mean_train,var_train,[100,100],true);