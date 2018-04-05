%EMPIRICAL NULL SCRIPT
%Takes a sample from a z image
%Plots a histogram of the z statistics before any empirical null correction
%Does the empirical null correction and print results

clc;
close all;
clearvars;

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

%define the coordinates of the subsample
row_subsample = 500:699; %array of row indicies
col_subsample = 500:699; %array of column indicies

%FIGURE
%Plot the z image with a rectangle highlighting the subsample
fig = LatexFigure.sub();
image_plot = ImagescSignificant(z_image);
image_plot.plot();
hold on;
rectangle('Position',[col_subsample(1), row_subsample(1), col_subsample(end)-col_subsample(1)+1, row_subsample(end)-row_subsample(1)+1],'EdgeColor','r','LineStyle','--');
ax = gca;
clim_for_all = ax.CLim; 
saveas(fig,fullfile('reports','figures','inference','alt_empirical_z_image.eps'),'epsc');

%get the subsample of z statistics and do BH multiple hypothesis testing
z_sample = z_image(row_subsample, col_subsample);
z_tester = ZTester(z_sample);
%estimate the empirical null
z_tester.estimateNull();
mu_0 = z_tester.mean_null; %get the empirical null mean
sigma_0 = sqrt(z_tester.var_null); %get the empirical null std
density_estimate = z_tester.density_estimator; %get the density estimator
x_plot = linspace(min(min(z_sample)),max(max(z_sample)),500); %define what values of x to plot the density estimate
f_hat = numel(z_sample)*density_estimate.getDensityEstimate(x_plot); %get the freqency density estimate

%Do hypothesis test, corrected using the empirical null
z_tester.doTest();
%get the critical boundary
z_bh_empirical = z_tester.getZCritical();

%FIGURE
%Plot the histogram of z statistics
%Also plot the empirical null BH critical boundary
fig = LatexFigure.sub();
z_tester.plotHistogram();
hold on;
z_tester.plotCritical();
ylabel('frequency density');
xlabel('z');
hold on;
legend('z histogram','critical','Location','northeast');
saveas(fig,fullfile('reports','figures','inference','alt_empirical_z_histo_null.eps'),'epsc');

%FIGURE
%Plot the frequency density estimate along with the empirical null mean and std
fig = LatexFigure.sub();
plot(x_plot,f_hat); %plot density estimate
hold on;
z_tester.plotCritical();
plot([mu_0,mu_0],[0,numel(z_sample)*density_estimate.getDensityEstimate(mu_0)],'k--'); %plot mode
%get the value of the density estimate at mu +/- sigma
f_at_sigma_1 = mean(density_estimate.getDensityEstimate([mu_0-sigma_0,mu_0+sigma_0]));
%plot line and arrows to represent 2 std
plot([mu_0-sigma_0,mu_0+sigma_0],numel(z_sample)*[f_at_sigma_1,f_at_sigma_1],'k--');
scatter(mu_0-sigma_0,numel(z_sample)*f_at_sigma_1,'k<','filled');
scatter(mu_0+sigma_0,numel(z_sample)*f_at_sigma_1,'k>','filled');
ylabel('frequency density');
xlabel('z');
hold on;
legend('density estimate','critical','Location','northeast');
saveas(fig,fullfile('reports','figures','inference','alt_empirical_z_parzen.eps'),'epsc');

%FIGURE
%plot the p values in order
%also plot the BH critical boundary
fig = LatexFigure.main();
z_tester.plotPValues();
legend('p values','critical','Location','northwest');
saveas(fig,fullfile('reports','figures','inference','alt_empirical_z_p_values.eps'),'epsc');

fig = LatexFigure.main();
image_plot = ImagescSignificant(z_image(row_subsample, col_subsample));
image_plot.addSigPixels(z_tester.sig_image);
image_plot.plot();
ax = gca;
ax.CLim = clim_for_all;
saveas(fig,fullfile('reports','figures','inference','alt_empirical_z_image_2.eps'),'epsc');