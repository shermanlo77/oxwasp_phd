clc;
close all;
clearvars;

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

fig = LatexFigure.sub();
image_plot = ImagescSignificant(z_image);
image_plot.plot();
hold on;
rectangle('Position',[400, 1100, 200, 200],'EdgeColor','r','LineStyle','--');
saveas(fig,fullfile('reports','figures','inference','sub_z_image.eps'),'epsc');

z_sample = reshape(z_image(1100:1299, 400:599),[],1);
z_tester = ZTester(z_sample);
z_tester.doTest();
z_bh = z_tester.getZCritical();
file_id = fopen(fullfile('reports','figures','inference','sub_boundary.txt'),'w');
fprintf(file_id,'%.2f',z_bh(2));
fclose(file_id);

fig = LatexFigure.sub();
ax = gca;
histogram_custom(z_sample);
ylabel('frequency density');
xlabel('z stat');
hold on;
saveas(fig,fullfile('reports','figures','inference','sub_z_histo_nocritical.eps'),'epsc');
plot([z_bh(1),z_bh(1)],ax.YLim,'r--');
plot([z_bh(2),z_bh(2)],ax.YLim,'r--');
legend('z histogram','critical boundary','Location','southeast');
saveas(fig,fullfile('reports','figures','inference','sub_z_histo.eps'),'epsc');

density_estimate = Parzen(z_sample);
x_plot = linspace(min(z_sample),max(z_sample),500);
f_hat = numel(z_sample)*density_estimate.getDensityEstimate(x_plot);
fig = LatexFigure.sub();
plot(x_plot,f_hat);
ylabel('frequency density');
xlabel('z stat');
saveas(fig,fullfile('reports','figures','inference','sub_z_parzen.eps'),'epsc');