clc;
clearvars;
close all;

block_data = AbsBlock_Sep16_120deg();

n_reference = block_data.getNReference();
reference_mean_var_array = zeros(2,n_reference);
reference_var_error = zeros(2,n_reference);
reference_mean_error = zeros(2,n_reference);
quantile_p = [normcdf(-1),0.5,normcdf(1)];
for i_reference = 1:n_reference
    reference_stack = block_data.reference_scan_array(i_reference).loadImageStack();
    reference_mean = reshape(mean(reference_stack,3),[],1);
    reference_var = reshape(var(reference_stack,[],3),[],1);
    
    mean_quantile = quantile(reference_mean,quantile_p);
    var_quantile = quantile(reference_var,quantile_p);
    
    reference_mean_var_array(:,i_reference) = [mean_quantile(2);var_quantile(2)];
    reference_mean_error(:,i_reference) = [mean_quantile(3);mean_quantile(1)] - mean_quantile(2);
    reference_var_error(:,i_reference) = [var_quantile(3);var_quantile(1)] - var_quantile(2);
    
end

% block_data.addDefaultShadingCorrector();
% block_data.addShadingCorrector(ShadingCorrector(),1:block_data.getNReference());
image_stack = block_data.loadImageStack();

mean_stack = mean(image_stack,3);
var_stack = var(image_stack,[],3);

mean_stack = reshape(mean_stack,[],1);
var_stack = reshape(var_stack,[],1);

segmentation = reshape(block_data.getSegmentation(),[],1);
mean_stack = mean_stack(segmentation);
var_stack = var_stack(segmentation);

figure;
hist3Heatmap(mean_stack, var_stack, [100,100], true);
colorbar;
xlabel('mean (arb. unit)');
ylabel('variance (arb. unit^2)');
hold on;
for i_ref = 1:n_reference
    errorbar(reference_mean_var_array(1,:),reference_mean_var_array(2,:),-reference_var_error(2,:),reference_var_error(1,:),-reference_mean_error(2,:),reference_mean_error(1,:),'r','LineStyle','none');
end
