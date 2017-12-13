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
    z_tester.doTest();
    p0 = z_tester.estimateP0();
    mean_null = z_tester.mean_null;
    std_null = z_tester.std_null;
    [mean_alt, std_alt] = z_tester.estimateAlternative(100);
    z_critical = z_tester.getZCritical();
    
    if i == 1
        mixture_gaussian = gmdistribution.fit(reshape(z_sub,[],1),2,'SharedCov',false);
        save('results/subregion1.mat','mixture_gaussian');
    end
    
    disp(strcat('p_0 = ',num2str(p0)));
    disp(strcat('Power = ',num2str(z_tester.estimatePower())));
    disp(strcat('Tail Power = ',num2str(z_tester.estimateTailPower(z_sub_plot(1),z_sub_plot(end),numel(z_sub_plot)))));
    disp(strcat('Local Power = ',num2str(z_tester.estimateLocalPower(z_sub_plot(1),z_sub_plot(end),numel(z_sub_plot)))));
    
    figure;
    histogram(z_sub,'Normalization','CountDensity','DisplayStyle','stairs');
    hold on;
    plot(z_sub_plot,m*z_tester.density_estimator.getDensityEstimate(z_sub_plot));
    plot(z_sub_plot,p0*m*normpdf(z_sub_plot,z_tester.mean_null,z_tester.std_null));
    plot(z_sub_plot,(1-p0)*m*z_tester.estimateH1Density(z_sub_plot));
    plot([z_critical(1),z_critical(1)],[0,m*normpdf(0)],'r-','LineWidth',2);
    plot([z_critical(2),z_critical(2)],[0,m*normpdf(0)],'r-','LineWidth',2);
    legend('histogram','estimated density','null density','non-null density');
    ylabel('frequency density');
    
    figure;
    yyaxis left;
    plot(z_sub_plot,z_tester.estimateLocalFdr(z_sub_plot));
    hold on;
    plot(z_sub_plot,z_tester.estimateTailFdr(z_sub_plot));
    ylabel('fdr');
    yyaxis right;
    plot(z_sub_plot,p0*m*normpdf(z_sub_plot,z_tester.mean_null,z_tester.std_null));
    plot(z_sub_plot,(1-p0)*m*z_tester.estimateH1Density(z_sub_plot));
    plot([z_critical(1),z_critical(1)],[0,m*normpdf(0)],'r-','LineWidth',2);
    plot([z_critical(2),z_critical(2)],[0,m*normpdf(0)],'r-','LineWidth',2);
    ylabel('frequency density');
    xlabel('z statistic');
    legend('local fdr','tail fdr','null density','non-null density');

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


factor_array = [0.9, 1.144, 2, 3.33]; %array of fudge factors
for i_factor = 1:numel(factor_array)
    grid_tester = GridTester(z_image, 200, 200, [0;0]);

    grid_tester.setDensityEstimationFudgeFactor(factor_array(i_factor));
    grid_tester.doTest(1000);
    fig = figure;
    imagesc(test);
    colorbar;
    hold on;
    colorbar;
    [critical_y, critical_x] = find(grid_tester.local_sig);
    scatter(critical_x, critical_y,'r.');
    colorbar;
    for i_row = 2:(grid_tester.n_row)
        corner_cood = grid_tester.getGridCoordinates(i_row, 1);
        plot([0,2000],[corner_cood(1),corner_cood(1)],'k');
    end
    for i_col = 2:(grid_tester.n_col)
        corner_cood = grid_tester.getGridCoordinates(1, i_col);
        plot([corner_cood(2),corner_cood(2)],[0,2000],'k');
    end
    fig.CurrentAxes.XTick = [];
    fig.CurrentAxes.YTick = [];
    
    z_tester = grid_tester.z_tester_array{7,8};
    z_critical = z_tester.getZCritical();
    z_vector = reshape(z_tester.z_image,[],1);
    m = sum(~isnan(z_vector));
    z_sub_plot = linspace(min(z_vector),max(z_vector),1000);
    figure;
    histogram(z_vector,'Normalization','CountDensity','DisplayStyle','stairs');
    hold on;
    plot(z_sub_plot,m*z_tester.density_estimator.getDensityEstimate(z_sub_plot));
    plot(z_sub_plot,p0*m*normpdf(z_sub_plot,z_tester.mean_null,z_tester.std_null));
    plot([z_critical(1),z_critical(1)],[0,m*normpdf(0)],'r-','LineWidth',2);
    plot([z_critical(2),z_critical(2)],[0,m*normpdf(0)],'r-','LineWidth',2);
    legend('histogram','estimated density','null density');
    ylabel('frequency density');
    
end


translation_array = [0.00, 0.50, 0.00, 0.50, 0.25, 0.75, 0.00, 0.00, 0.25, 0.75, 0.75, 0.25
                     0.00, 0.00, 0.50, 0.50, 0.00, 0.00, 0.25, 0.75, 0.25, 0.75, 0.25, 0.75];
n_translation_array = [1,3,4,8,12];
grid_tester = MultiGridTester(z_image, 200, 200, translation_array);
grid_tester.doTest(1000);
for i_trans_series = 1:numel(n_translation_array)
    
    n_translation = n_translation_array(i_trans_series);
    
    fig = figure;
    imagesc(test);
    colorbar;
    hold on;
    colorbar;
    [critical_y, critical_x] = find(grid_tester.majorityLocalVoteIndex(1:n_translation));
    scatter(critical_x, critical_y,'r.');
    fig.CurrentAxes.XTick = [];
    fig.CurrentAxes.YTick = [];
    
    fig = figure;
    imagesc(test);
    colorbar;
    hold on;
    colorbar;
    [critical_y, critical_x] = find(grid_tester.majorityCombinedVoteIndex(1:n_translation));
    scatter(critical_x, critical_y,'r.');
    fig.CurrentAxes.XTick = [];
    fig.CurrentAxes.YTick = [];
    
    fig = figure;
    imagesc(log10(mean(grid_tester.p_image_array(:,:,1:n_translation),3)));
    colorbar;
    hold on;
    colorbar;
    fig.CurrentAxes.XTick = [];
    fig.CurrentAxes.YTick = [];

%     for i_translation = 1:n_translation
%         
%         shift = translation_array(:,i_translation);
%         n_row = ceil((2000+shift(1))/200);
%         n_col = ceil((2000+shift(2))/200);
%         
%         fig = figure;
%         imagesc(test);
%         colorbar;
%         hold on;
%         colorbar;
%         [critical_y, critical_x] = find(grid_tester.combined_sig_array(:,:,i_translation));
%         scatter(critical_x, critical_y,'r.');
%         colorbar;
%         for i_row = 2:(n_row)
%             corner_cood = GridTester.STATIC_getGridCoordinates(2000, 2000, i_row, 1, 200, 200, shift.*[200; 200]);
%             plot([0,2000],[corner_cood(1),corner_cood(1)],'k');
%         end
%         for i_col = 2:(n_col)
%             corner_cood = GridTester.STATIC_getGridCoordinates(2000, 2000, 1, i_col, 200, 200, shift.*[200; 200]);
%             plot([corner_cood(2),corner_cood(2)],[0,2000],'k');
%         end
%         fig.CurrentAxes.XTick = [];
%         fig.CurrentAxes.YTick = [];
% 
%         fig = figure;
%         imagesc(test);
%         colorbar;
%         hold on;
%         colorbar;
%         [critical_y, critical_x] = find(grid_tester.local_sig_array(:,:,i_translation));
%         scatter(critical_x, critical_y,'r.');
%         colorbar;
%         for i_row = 2:(n_row)
%             corner_cood = GridTester.STATIC_getGridCoordinates(2000, 2000, i_row, 1, 200, 200, shift.*[200; 200]);
%             plot([0,2000],[corner_cood(1),corner_cood(1)],'k');
%         end
%         for i_col = 2:(n_col)
%             corner_cood = GridTester.STATIC_getGridCoordinates(2000, 2000, 1, i_col, 200, 200, shift.*[200; 200]);
%             plot([corner_cood(2),corner_cood(2)],[0,2000],'k');
%         end
%         fig.CurrentAxes.XTick = [];
%         fig.CurrentAxes.YTick = [];
%     end
end

fig = figure;
imagesc(test);
colorbar;
hold on;
colorbar;
[critical_y, critical_x] = find(sum(sig_combine_array,3)>=3);
scatter(critical_x, critical_y,'r.');
colorbar;
fig.CurrentAxes.XTick = [];
fig.CurrentAxes.YTick = [];

fig = figure;
imagesc(test);
colorbar;
hold on;
colorbar;
[critical_y, critical_x] = find(sum(sig_local_array,3)>=3);
scatter(critical_x, critical_y,'r.');
colorbar;
fig.CurrentAxes.XTick = [];
fig.CurrentAxes.YTick = [];

% % chi_squared = -2*sum(log(p_value_array),3);
% % chi_squared_p = chi2cdf(chi_squared,2*n_translation,'upper');
% chi_squared_p = median(p_value_array,3);
% p_tester = PTester(chi_squared_p,2*normcdf(-2));
% p_tester.doTest();
% 
% fig = figure;
% imagesc(log10(chi_squared_p));
% colorbar;
% fig.CurrentAxes.XTick = [];
% fig.CurrentAxes.YTick = [];
% 
% fig = figure;
% imagesc(test);
% hold on;
% [critical_y, critical_x] = find(p_tester.sig_image);
% scatter(critical_x, critical_y,'r.');
% colorbar;
% fig.CurrentAxes.XTick = [];
% fig.CurrentAxes.YTick = [];