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

%put the z image in a tester
z_tester = ZTester(z_image);
%do statistics on the z statistics
z_tester.doTest();
    
%plot the phantom and aRTist image
fig = LatexFigure.sub();
phantom_plot = ImagescSignificant(test);
phantom_plot.plot();
ax = gca;
ax.CLim = [2.2E4,5.5E4];
saveas(fig,fullfile('reports','figures','inference','scan.eps'),'epsc');

fig = LatexFigure.sub();
phantom_plot = ImagescSignificant(aRTist);
phantom_plot.plot();
ax = gca;
ax.CLim = [2.2E4,5.5E4];
saveas(fig,fullfile('reports','figures','inference','aRTist.eps'),'epsc');

fig = LatexFigure.sub();
image_plot = ImagescSignificant(z_image);
image_plot.plot();
saveas(fig,fullfile('reports','figures','inference','z_image.eps'),'epsc');

fig = LatexFigure.sub();
image_plot = ImagescSignificant(-log10(z_tester.p_image));
image_plot.plot();
saveas(fig,fullfile('reports','figures','inference','logp.eps'),'epsc');

%plot the phantom scan with critical pixels highlighted
fig = LatexFigure.main;
image_plot = ImagescSignificant(test);
image_plot.addSigPixels(z_tester.sig_image);
image_plot.plot();
saveas(fig,fullfile('reports','figures','inference','sig_pixels.eps'),'epsc');

%histogram
fig = LatexFigure.main(z_tester.figureHistCritical());
saveas(fig,fullfile('reports','figures','inference','z_histo.eps'),'epsc');

z_critical = z_tester.getZCritical();
z_critical = z_critical(2);
file_id = fopen(fullfile('reports','figures','inference','z_critical.txt'),'w');
fprintf(file_id,'%.2f',z_critical);
fclose(file_id);

tic;
convolution = EmpericalConvolution(z_image,20, 20, [200,200]);
convolution.estimateNull();
convolution.setMask(segmentation);
convolution.doTest();
toc;

fig = figure();
fig.Position(3:4) = [420,315];
image_plot = ImagescSignificant(z_image-convolution.mean_null);
image_plot.plot();

convolution.z_tester.figureHistCritical();

fig = figure;
fig.Position(3:4) = [420,315];
image_plot = ImagescSignificant(-log10(convolution.p_image));
image_plot.plot();

fig = figure;
fig.Position(3:4) = [420,315];
image_plot = ImagescSignificant(test);
image_plot.setDilateSize(1);
image_plot.addSigPixels(convolution.sig_image);
image_plot.plot();


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
