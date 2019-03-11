%SET UP INFERENCE EXAMPLE SCRIPT
%This script sets up the z_image for the AbsBlock_Sep16_120deg() dataset
%19 images are used to train the variance-mean relationship
%1 (held out) image is then compared with the aRTist simulation, obtaining a z image

%set random seed
rand_stream = RandStream('mt19937ar','Seed',uint32(3538096789));

%load data and add shading correction
block_data = AbsBlock_Sep16_120deg();
block_data.addDefaultShadingCorrector();

%get random permutation for each image
index = rand_stream.randperm(block_data.n_sample);
n_test = 1;
n_train = block_data.n_sample - n_test;
training_index = index(1:n_train);
test_index = index((n_train+1):(n_train+n_test));

%get a phanton image and aRTist image
aRTist = block_data.getShadingCorrectedARTistImage(ShadingCorrector(),1:block_data.reference_white);

%get the segmentation image
segmentation = block_data.getSegmentation();
%get the number of segmented images
n_pixel = sum(sum(segmentation));

%get the training images
training_stack = block_data.loadImageStack(training_index);
%segment the image
training_stack = reshape(training_stack,block_data.area,n_train);
training_stack = training_stack(reshape(segmentation,[],1),:);
%get the segmented mean and variance greyvalue
training_mean = mean(training_stack,2);
training_var = var(training_stack,[],2);

%train glm using the training set mean and variance
model = GlmGamma(1,IdentityLink());
model.setShapeParameter((n_train-1)/2);
model.train(training_mean,training_var);

%predict variance given aRTist
var_predict = reshape(model.predict(reshape(aRTist,[],1)),block_data.height, block_data.width);

%get the test images
test = block_data.loadImageStack(test_index);

%get the z statistic
z_image = (test - aRTist)./sqrt(var_predict);
%set non segmented pixels to be nan
z_image(~segmentation) = nan;
%find the number of non-nan pixels
m = sum(sum(~isnan(z_image)));