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
n_train = block_data.n_sample - 2;
n_test = 1;
n_calibrate = 1;
training_index = index(1:n_train);
test_index = index((n_train+1):(n_train+n_test));
calibrate_index = index((n_train+n_test+1):end);

%get a phanton image and aRTist image
phantom = mean(block_data.loadImageStack(calibrate_index),3);
aRTist = block_data.getShadingCorrectedARTistImage(ShadingCorrector(),1:block_data.reference_white);

%get the segmentation image
segmentation = block_data.getSegmentation();
%get the number of segmented images
n_pixel = sum(sum(segmentation));

phantom_vector = reshape(phantom(segmentation),[],1);
aRTist_vector = reshape(aRTist(segmentation),[],1);
d = phantom_vector - aRTist_vector;

%plot aRTist vs phantom greyvalue as a histogram heatmap
figure;
hist3Heatmap(phantom_vector,aRTist_vector,[300,300],true);
hold on;
%get the min and max greyvalue
min_grey = min([min(min(phantom)),min(min(aRTist))]);
max_grey = max([max(max(phantom)),max(max(aRTist))]);
%plot straight line with gradient 1
plot([min_grey,max_grey],[min_grey,max_grey],'r');
%label axis
colorbar;
xlabel('phantom greyvalue (arb. unit)');
ylabel('aRTist greyvalue (arb. unit)');

%plot phantom - aRTist vs aRTist greyvalue as a histogram heatmap
figure;
hist3Heatmap(aRTist_vector,d,[100,100],true);
hold on;
%label axis
colorbar;
xlabel('aRTist greyvalue (arb. unit)');
ylabel('difference in greyvalue (arb. unit)');

model = MeanVar_kNN(1E5);
model.train(aRTist_vector,d);
aRTist_plot = (min(aRTist_vector):max(aRTist_vector))';
hold on;
plot(aRTist_plot,model.predict(aRTist_plot),'-r');

d_predict = model.predict(reshape(aRTist,[],1));
aRTist = aRTist + reshape(d_predict,block_data.height,block_data.width);
aRTist_plot = (min(min(aRTist)):max(max(aRTist_vector)))';

aRTist_vector = reshape(aRTist(segmentation),[],1);
d = phantom_vector - aRTist_vector;

%plot the phantom and aRTist image
figure;
imagesc(phantom);
colorbar;
figure;
imagesc(aRTist);
colorbar;

%plot aRTist vs phantom greyvalue as a histogram heatmap
figure;
hist3Heatmap(phantom_vector,aRTist_vector,[300,300],true);
hold on;
%get the min and max greyvalue
min_grey = min([min(min(phantom)),min(min(aRTist))]);
max_grey = max([max(max(phantom)),max(max(aRTist))]);
%plot straight line with gradient 1
plot([min_grey,max_grey],[min_grey,max_grey],'r');
%label axis
colorbar;
xlabel('phantom greyvalue (arb. unit)');
ylabel('aRTist greyvalue (arb. unit)');

%plot phantom - aRTist vs aRTist greyvalue as a histogram heatmap
figure;
hist3Heatmap(aRTist_vector,d,[100,100],true);
%label axis
colorbar;
xlabel('aRTist greyvalue (arb. unit)');
ylabel('difference in greyvalue (arb. unit)');

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
model = MeanVar_GLM((n_train-1)/2,1,LinkFunction_Identity());
model.train(training_mean,training_var);

%predict variance given aRTist
var_predict = reshape(model.predict(reshape(aRTist,[],1)),block_data.height, block_data.width);

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
    %set non segmented pixels to be nan
    z_image(~segmentation) = nan;
    %find the number of non-nan pixels
    m = sum(sum(~isnan(z_image)));
    
    %put the z image in a tester
    z_tester = ZTester(z_image);
    %do statistics on the z statistics
    z_tester.getPValues();
    z_tester.doTest();

    fig = figure;
    imagesc_truncate(z_image);
    colorbar;
    fig.CurrentAxes.XTick = [];
    fig.CurrentAxes.YTick = [];
    
    figure;
    imagesc(log10(z_tester.p_image));
    colorbar;
    
    %histogram
    z_vector = reshape(z_image,[],1);
    z_vector(isnan(z_vector)) = [];
    z_plot = linspace(min(z_vector),max(z_vector),1000);
    figure;
    histogram(z_vector,'Normalization','CountDensity','DisplayStyle','stairs');
    hold on;
    plot(z_plot,normpdf(z_plot)*m,'--');
    plot([-norminv(1-z_tester.size_corrected/2),-norminv(1-z_tester.size_corrected/2)],[0,m*normpdf(0)],'r-','LineWidth',2);
    plot([norminv(1-z_tester.size_corrected/2),norminv(1-z_tester.size_corrected/2)],[0,m*normpdf(0)],'r-','LineWidth',2);
    xlabel('z statistic');
    ylabel('frequency density');
    legend('histogram','N(0,1)','critical boundary');

    %plot the phantom scan with critical pixels highlighted
    [critical_y, critical_x] = find(z_tester.sig_image);
    fig = figure;
    imagesc(test);
    hold on;
    scatter(critical_x, critical_y,'r.');
    colorbar;
    fig.CurrentAxes.XTick = [];
    fig.CurrentAxes.YTick = [];
    
end

row_array = {547:700, 522:708, 1800:1910, 1060:1260};
col_array = {580:708, 766:997, 852:1020, 816:1010};
z_plot = linspace(-5,5,100);

for i = 1:numel(col_array)

    col_index = col_array{i};
    row_index = row_array{i};

    z_sub = z_image(row_index, col_index);
    m = sum(sum(~isnan(z_sub)));
    z_tester = ZTester(z_sub);
    
    z_sub_plot = linspace(min(min(z_sub)),max(max(z_sub)),100);
    
    z_tester.estimateNull(100);
    z_tester.getPValues();
    z_tester.doTest();
    z_critical = z_tester.getZCritical();
    
    figure;
    histogram(z_sub,'Normalization','CountDensity','DisplayStyle','stairs');
    hold on;
    plot(z_sub_plot,m*z_tester.density_estimator.getDensityEstimate(z_sub_plot));
    plot(z_sub_plot,m*normpdf(z_sub_plot,z_tester.mean_null,z_tester.std_null));
    plot([z_critical(1),z_critical(1)],[0,m*normpdf(0)],'r-','LineWidth',2);
    plot([z_critical(2),z_critical(2)],[0,m*normpdf(0)],'r-','LineWidth',2);
    xlabel('z statistic');
    ylabel('frequency density');
    legend('histogram','estimated density','null density');

    fig = figure;
    imagesc_truncate(z_image);
    colorbar;
    hold on;
    plot([col_index(1),col_index(end)],[row_index(1),row_index(1)],'r','LineWidth',2);
    plot([col_index(1),col_index(end)],[row_index(end),row_index(end)],'r','LineWidth',2);
    plot([col_index(1),col_index(1)],[row_index(1),row_index(end)],'r','LineWidth',2);
    plot([col_index(end),col_index(end)],[row_index(1),row_index(end)],'r','LineWidth',2);
    fig.CurrentAxes.XTick = [];
    fig.CurrentAxes.YTick = [];
    
    fig = figure;
    imagesc_truncate(z_sub);
    hold on;
    colorbar;
    [critical_y, critical_x] = find(z_tester.sig_image);
    scatter(critical_x, critical_y,'r.');
    colorbar;
    fig.CurrentAxes.XTick = [];
    fig.CurrentAxes.YTick = [];
    
end

n_grid = 10;
sub_region_size = 2000 / n_grid;

fig = figure;
imagesc(test);
hold on;
colorbar;
fig.CurrentAxes.XTick = [];
fig.CurrentAxes.YTick = [];
for i = 1:(n_grid-1)
    plot([0,2000],[i*sub_region_size,i*sub_region_size],'k');
    plot([i*sub_region_size,i*sub_region_size],[0,2000],'k');
end

z_null_image = test;
z_null_image(:) = nan;
z_tester_grid = cell(n_grid,n_grid);
sig_null_image = test;
sig_null_image(:) = false;
for i_col = 1:n_grid
    for i_row = 1:n_grid
        z_tester_grid{i_row,i_col} = ZTester(z_image( ((i_row-1)*sub_region_size+1):(i_row*sub_region_size) , ((i_col-1)*sub_region_size+1):(i_col*sub_region_size) ) );
        %z_tester_grid{i_row,i_col}.setDensityEstimationParameter(0.4);
        z_tester_grid{i_row,i_col}.estimateNull(1000);
        z_null_image( ((i_row-1)*sub_region_size+1):(i_row*sub_region_size) , ((i_col-1)*sub_region_size+1):(i_col*sub_region_size) ) = z_tester_grid{i_row,i_col}.getZCorrected();
        z_tester_grid{i_row,i_col}.getPValues();
        z_tester_grid{i_row,i_col}.doTest();
        sig_null_image( ((i_row-1)*sub_region_size+1):(i_row*sub_region_size) , ((i_col-1)*sub_region_size+1):(i_col*sub_region_size) ) = z_tester_grid{i_row,i_col}.sig_image;
    end
end

z_tester = ZTester(z_null_image);
z_tester.getPValues();
z_tester.doTest();

fig = figure;
imagesc(log10(z_tester.p_image));
colorbar;
fig.CurrentAxes.XTick = [];
fig.CurrentAxes.YTick = [];

fig = figure;
imagesc(z_null_image);
colorbar;
fig.CurrentAxes.XTick = [];
fig.CurrentAxes.YTick = [];

fig = figure;
imagesc(test);
colorbar;
hold on;
colorbar;
[critical_y, critical_x] = find(z_tester.sig_image);
scatter(critical_x, critical_y,'r.');
colorbar;
fig.CurrentAxes.XTick = [];
fig.CurrentAxes.YTick = [];

fig = figure;
imagesc(test);
colorbar;
hold on;
colorbar;
[critical_y, critical_x] = find(sig_null_image);
scatter(critical_x, critical_y,'r.');
colorbar;
fig.CurrentAxes.XTick = [];
fig.CurrentAxes.YTick = [];