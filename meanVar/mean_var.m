%MEAN VAR
%Plots the mean and variance frequency density heatmap for the data containing reference images
%Also plots the quantiles of the mean and variance of the reference images

clc;
clearvars;
close all;

hist_plot = Hist3Heatmap();

%for each data set
for i_data = 1:4
    
    %get the data
    switch i_data
        case 1
            block_data = AbsBlock_July16_30deg();
            block_data.addDefaultShadingCorrector();
            data_name = 'AbsBlock_July16_30deg';
        case 2
            block_data = AbsBlock_July16_120deg();
            block_data.addDefaultShadingCorrector();
            data_name = 'AbsBlock_July16_120deg';
        case 3
            block_data = AbsBlock_Sep16_30deg();
            block_data.addDefaultShadingCorrector();
            data_name = 'AbsBlock_Sep16_30deg';
        case 4
            block_data = AbsBlock_Sep16_120deg();
            block_data.addDefaultShadingCorrector();
            data_name = 'AbsBlock_Sep16_120deg';
    end
    
    %get the number of reference images
    n_reference = block_data.getNReference();
    
    %declare array for storing the reference images mean and variance for power
        %dim 1: [mean, variance]
        %dim 2: for each power
    reference_mean_var_array = zeros(2,n_reference);
    %declare array for storing the error bars for the variance and mean
        %dim 1: [upper,lower] size of error bar, lower value is negative
        %dim 2: for each power
    reference_var_error = zeros(2,n_reference);
    reference_mean_error = zeros(2,n_reference);
    
    %quantiles used for the error bars
    quantile_p = [normcdf(-1),0.5,normcdf(1)];
    
    %for each reference image
    for i_reference = 1:n_reference
        
        %get the stack of reference images
        reference_stack = block_data.reference_scan_array(i_reference).loadImageStack();
        %get the mean and variance for each pixel
        reference_mean = reshape(mean(reference_stack,3),[],1);
        reference_var = reshape(var(reference_stack,[],3),[],1);

        %get the quantiles of the mean and variances
        mean_quantile = quantile(reference_mean,quantile_p);
        var_quantile = quantile(reference_var,quantile_p);

        %store the median mean and variance
        reference_mean_var_array(:,i_reference) = [mean_quantile(2);var_quantile(2)];
        %store the quantiles of the mean and variance
        reference_mean_error(:,i_reference) = [mean_quantile(3);mean_quantile(1)] - mean_quantile(2);
        reference_var_error(:,i_reference) = [var_quantile(3);var_quantile(1)] - var_quantile(2);

    end

    %get the mean and variance of each pixel in the scan
    mean_var_estimator = MeanVarianceEstimator(block_data);
    [mean_stack,var_stack] = mean_var_estimator.getMeanVar(1:block_data.n_sample);

    %heatmap plot the mean and variance frequency density
    fig = LatexFigure.main();
    hist_plot.plot(mean_stack, var_stack);
    xlabel('mean (arb. unit)');
    ylabel('variance (arb. unit)');
    hold on;
    %for each reference image, plot the error bars of each reference image
    for i_ref = 1:n_reference
        errorbar(reference_mean_var_array(1,:),reference_mean_var_array(2,:),-reference_var_error(2,:),reference_var_error(1,:),-reference_mean_error(2,:),reference_mean_error(1,:),'r','LineStyle','none');
    end
    %save the figure
    saveas(fig,fullfile('reports','figures','meanVar',strcat('meanVar',data_name,'.eps')),'epsc');
end