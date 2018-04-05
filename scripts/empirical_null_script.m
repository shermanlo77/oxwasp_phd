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
row_subsample = 1100:1299; %array of row indicies
col_subsample = 400:599; %array of column indicies

%FIGURE
%Plot the z image with a rectangle highlighting the subsample
fig = LatexFigure.sub();
image_plot = ImagescSignificant(z_image);
image_plot.plot();
hold on;
rectangle('Position',[col_subsample(1), row_subsample(1), col_subsample(end)-col_subsample(1)+1, row_subsample(end)-row_subsample(1)+1],'EdgeColor','r','LineStyle','--');
saveas(fig,fullfile('reports','figures','inference','sub_z_image.eps'),'epsc');

%get the subsample of z statistics and do BH multiple hypothesis testing
z_sample = reshape(z_image(row_subsample, col_subsample),[],1);
z_tester = ZTester(z_sample);
z_tester.doTest();
z_bh = z_tester.getZCritical();

%SAVE VALUE
%Save the critical boundary for the BH procedure
file_id = fopen(fullfile('reports','figures','inference','sub_boundary.txt'),'w');
fprintf(file_id,'%.2f',z_bh(2));
fclose(file_id);

%FIGURE
%Plot the histogram of the z statistics
%Plot the histogram of the z statistics with the BH critical boundary
fig = LatexFigure.sub();
z_tester.plotHistogram();
ylabel('frequency density');
xlabel('z stat');
hold on;
saveas(fig,fullfile('reports','figures','inference','sub_z_histo_nocritical.eps'),'epsc');
z_tester.plotCritical();
legend('z histogram','critical','Location','southeast');
saveas(fig,fullfile('reports','figures','inference','sub_z_histo.eps'),'epsc');

%estimate the empirical null
z_tester.estimateNull();
mu_0 = z_tester.mean_null; %get the empirical null mean
sigma_0 = sqrt(z_tester.var_null); %get the empirical null std
density_estimate = z_tester.density_estimator; %get the density estimator
x_plot = linspace(min(z_sample),max(z_sample),500); %define what values of x to plot the density estimate
f_hat = numel(z_sample)*density_estimate.getDensityEstimate(x_plot); %get the freqency density estimate

%FIGURE
%Plot the frequency density estimate along with the empirical null mean and std
fig = LatexFigure.sub(); 
plot(x_plot,f_hat); %plot density estimate
hold on;
plot([mu_0,mu_0],[0,numel(z_sample)*density_estimate.getDensityEstimate(mu_0)],'k--'); %plot mode
%get the value of the density estimate at mu +/- sigma
f_at_sigma_1 = mean(density_estimate.getDensityEstimate([mu_0-sigma_0,mu_0+sigma_0]));
%plot line and arrows to represent 2 std
plot([mu_0-sigma_0,mu_0+sigma_0],numel(z_sample)*[f_at_sigma_1,f_at_sigma_1],'k--');
scatter(mu_0-sigma_0,numel(z_sample)*f_at_sigma_1,'k<','filled');
scatter(mu_0+sigma_0,numel(z_sample)*f_at_sigma_1,'k>','filled');
ylabel('frequency density');
xlabel('z stat');
saveas(fig,fullfile('reports','figures','inference','sub_z_parzen.eps'),'epsc');

%%%%SECTION FOR BOOTSTRAP
%It was found the std of the estimators are pretty much negliable
% n_bootstrap = 1000;
% mu_array = zeros(n_bootstrap,1);
% sigma_array = zeros(n_bootstrap,1);
% for i = 1:n_bootstrap
%     z_bootstrap = z_sample( rand_stream.randi(numel(z_sample),numel(z_sample),1) );
%     z_tester_bootstrap = ZTester(z_bootstrap);
%     z_tester.estimateNull();
%     mu_array(i) = z_tester.mean_null;
%     sigma_array(i) = sqrt(z_tester.var_null);
% end

%SAVE VALUE
%save the empirical null mean
file_id = fopen(fullfile('reports','figures','inference','sub_null_mu.txt'),'w');
fprintf(file_id,'%.2f',mu_0);
fclose(file_id);

%SAVE VALUE
%save the empirical null std
file_id = fopen(fullfile('reports','figures','inference','sub_null_sigma.txt'),'w');
fprintf(file_id,'%.2f',sigma_0);
fclose(file_id);

%Do hypothesis test, corrected using the empirical null
z_tester.doTest();
%get the critical boundary
z_bh_empirical = z_tester.getZCritical();

%SAVE VALUE
%Save the lower critical boundary for the empirical null BH procedure
file_id = fopen(fullfile('reports','figures','inference','sub_null_critical1.txt'),'w');
fprintf(file_id,'%.2f',z_bh_empirical(1));
fclose(file_id);

%SAVE VALUE
%Save the upper critical boundary for the empirical null BH procedure
file_id = fopen(fullfile('reports','figures','inference','sub_null_critical2.txt'),'w');
fprintf(file_id,'%.2f',z_bh_empirical(2));
fclose(file_id);

%SAVE VALUE
%Save the standarised critical boundary for the empirical null BH procedure
z_critical = norminv(1-z_tester.size_corrected/2);
file_id = fopen(fullfile('reports','figures','inference','sub_null_critical_zeta.txt'),'w');
fprintf(file_id,'%.2f',z_critical);
fclose(file_id);

%FIGURE
%Plot the histogram of z statistics
%Also plot the empirical null BH critical boundary
fig = LatexFigure.sub();
ax = gca;
z_tester.plotHistogram();
ylabel('frequency density');
xlabel('z stat');
hold on;
z_tester.plotCritical();
legend('z histogram','critical','Location','southeast');
saveas(fig,fullfile('reports','figures','inference','sub_z_histo_null.eps'),'epsc');

%FIGURE
%plot the p values in order
%also plot the BH critical boundary
fig = LatexFigure.sub();
z_tester.plotPValues();
legend('p values','critical','Location','northwest');
saveas(fig,fullfile('reports','figures','inference','sub_z_p_values.eps'),'epsc');