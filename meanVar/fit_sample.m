clc;
clearvars;
close all;

nbin = 100;

[sample_var,sample_mean] = getSampleMeanVar('/home/sherman/Documents/data/block',400);

shape_parameter = (100-1)/2;
model = MeanVar_GLM_canonical(shape_parameter);
model.train(sample_var,sample_mean,100);

x_plot = linspace(min(sample_mean),max(sample_mean),100);
[variance_prediction, up_error, down_error] = model.predict(x_plot');

plotHistogramHeatmap(sample_mean,sample_var,nbin);
colormap gray;
hold on;
plot(x_plot,variance_prediction,'r');
plot(x_plot,up_error,'r--');
plot(x_plot,down_error,'r--');