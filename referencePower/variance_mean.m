%VARIANCE MEAN SCRIPT
%Plots a histogram heat map of variance-mean pair
%of each pixel from each reference scan

clc;
clearvars;
close all;

%get the data
bgw_data = AbsBlock_Sep16_30deg();
n_reference = numel(bgw_data.reference_scan_array);

power_array = zeros(1,n_reference);
for i = 1:n_reference
    power_array(i) = bgw_data.reference_scan_array(i).power;
end

%apply shading correction

bgw_data.addShadingCorrector(ShadingCorrector(),1:n_reference);
bgw_data.turnOnRemoveDeadPixels();

% bgw_data.addShadingCorrector(ShadingCorrector(),[1,n_reference]);
% bgw_data.turnOnRemoveDeadPixels();

%declare array to store mean in mean_array and variance in var_array
mean_array = zeros(bgw_data.area,n_reference);
var_array = zeros(bgw_data.area,n_reference);

%for each reference scan
for i_ref = 1:n_reference
    
    %get the stack of images from this scan
    image_array = bgw_data.reference_scan_array(i_ref).loadImageStack();
    %get the mean and variance
    mean_image = mean(image_array,3);
    var_image = var(image_array,[],3);
    
    %append the mean and variance to mean_array and var_array
    mean_array(:, i_ref) = reshape(mean_image,[],1);
    var_array(:, i_ref) = reshape(var_image,[],1);
end

%plot histogram
figure;
hist3Heatmap(reshape(mean_array,[],1),reshape(var_array,[],1),[100,100],true);
colorbar;
xlabel('mean (arb. unit)');
ylabel('variance  (arb. unit^2)');

figure;
boxplot(mean_array,'position', power_array,'boxstyle','filled','medianstyle','target','outliersize',4,'symbol','o');
xlabel('Power (W)');
ylabel('Pixel mean (arb. unit)');