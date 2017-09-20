clc;
clearvars;
close all;

rng(uint32(3134830320), 'twister');

polynomial_order = 1;
link_function = LinkFunction_Identity();
n_plot = 100;
n_bootstrap = 1E3;

scan = AbsBlock_Sep16_30deg();
scan.addShadingCorrector(ShadingCorrector,1:scan.reference_white);
image_stack = scan.loadImageStack();
segmentation = scan.getSegmentation();

%get the number of segmented pixels
n_pixel = sum(sum(segmentation));

shape_parameter = (scan.n_sample-1)/2;

%load the images and reshape it to be a design matrix
image_stack = reshape(image_stack,scan.area,scan.n_sample);

%segment the design matrix
greyvalue_array = image_stack(segmentation,:);

sample_mean = mean(greyvalue_array,2);
sample_var = var(greyvalue_array,[],2);
model = MeanVar_GLM(shape_parameter,polynomial_order,link_function);
model.train(sample_mean,sample_var);

x_plot = (linspace(min(sample_mean),max(sample_mean),n_plot))';
[y_plot, up_error, down_error] = model.predict(x_plot);
figure;
hist3Heatmap(sample_mean,sample_var,[100,100],true);
hold on;
plot(x_plot,y_plot,'r');
plot(x_plot,up_error,'r--');
plot(x_plot,down_error,'r--');


y_array = zeros(n_plot, n_bootstrap);
y_predict = zeros(n_plot, n_bootstrap);

for i = 1:n_bootstrap
    index = randi([1,scan.n_sample],scan.n_sample,1);
    sample_mean = mean(greyvalue_array(:,index),2);
    sample_var = var(greyvalue_array(:,index),[],2);
    model = MeanVar_GLM(shape_parameter,polynomial_order,link_function);
    model.train(sample_mean,sample_var);
    y_predict(:,i) = model.predict(x_plot);
    
    for j = 1:n_plot
        [~,var_index] = min(abs(sample_mean - x_plot(j)));
        y_array(j,i) = sample_var(var_index);
    end
    
end

f = repmat(mean(y_array,2),1,n_bootstrap);

rss_plot = mean( (y_array - y_predict).^2,2);
mse_plot = mean( (y_predict - f).^2,2);
bias_plot = mean(y_predict,2) - mean(y_array,2);
var_plot = var(y_predict,[],2);
noise_plot = var(y_array,[],2);

figure;
plot(x_plot,rss_plot);
hold on;
plot(x_plot,noise_plot);
legend('RSS','\sigma^2');
xlabel('mean greyvalue (arb. unit)');
ylabel('statistic (arb. unit^2)');

figure;
plot(x_plot,mse_plot);
hold on;
plot(x_plot,var_plot);
plot(x_plot,bias_plot.^2);
legend('MSE','VAR','BIAS^2');
xlabel('mean greyvalue (arb. unit)');
ylabel('statistic (arb. unit^2)');

figure;
plot(x_plot,bias_plot);
xlabel('mean greyvalue (arb. unit)');
ylabel('bias (arb. unit)');