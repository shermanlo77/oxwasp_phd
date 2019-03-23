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

defect_simulator = DefectSimulator([block_data.height,block_data.width]);
defect_simulator.addPlane( (1E3/(sqrt(2)*1000))*[1;1]);
defect_simulator.addSinusoid(1E3, [750;750],0);

meanvar_index = index(1:(n_train-1));
test_index = index(n_train-1);
artist_index = index((n_train+1):end);

aRTist = mean(block_data.loadImageStack(artist_index),3);

fig = figure;
imagesc(aRTist);
colorbar;
fig.CurrentAxes.XTick = [];
fig.CurrentAxes.YTick = [];

%get the training images
training_stack = block_data.loadImageStack(meanvar_index);

for i = 1:numel(meanvar_index)
    training_stack(:,:,i) = defect_simulator.defectImage(training_stack(:,:,i));
end

%segment the image
training_stack = reshape(training_stack,block_data.area,n_train-1);
training_stack = training_stack(reshape(segmentation,[],1),:);
%get the segmented mean and variance greyvalue
training_mean = mean(training_stack,2);
training_var = var(training_stack,[],2);
figure;
hist3Heatmap(training_mean,training_var,[100,100],true);
xlabel('mean');
ylabel('variance');
colorbar;

%train glm using the training set mean and variance
model = GlmGamma(1,IdentityLink());
model.setShapeParameter((n_train-2)/2);
model.train(training_mean,training_var);

%predict variance given aRTist
var_predict = reshape(model.predict(reshape(aRTist,[],1)),block_data.height, block_data.width);

%get the test images
test_0 = block_data.loadImageStack(test_index);
test = defect_simulator.defectImage(test_0);

z_image = (test - aRTist)./sqrt(var_predict);
z_image(~segmentation) = nan;

convolution = EmpericalConvolution(z_image,20, 20, [200,200]);
convolution.setUseVarUniform(false);
convolution.estimateNull(1000);
convolution.setMask(segmentation);
convolution.doTest();

sig_0 = defect_simulator.sig_image;
sig_0(~segmentation) = 0;

convolution.z_tester.figureHistCritical();

figure;
image_plot = ImagescSignificant(-log10(convolution.p_image));
image_plot.plot();

figure;
image_plot = ImagescSignificant(test);
image_plot.plot();

figure;
image_plot.addSigPixels(convolution.sig_image);
image_plot.plot();

figure;
image_plot = ImagescSignificant(convolution.mean_null);
image_plot.plot();

figure;
image_plot = ImagescSignificant(sqrt(convolution.var_null));
image_plot.plot();
