%VARIANCE MEAN SCRIPT
%Plots a histogram heat map of variance-mean pair
%of each pixel from each reference scan

clc;
clearvars;
close all;

%get the data
bgw_data = AbsBlock_Sep16_30deg();
n_reference = numel(bgw_data.reference_scan_array);

%apply shading correction

bgw_data.addShadingCorrector(ShadingCorrector(),1:n_reference);
bgw_data.turnOnRemoveDeadPixels();

% bgw_data.addShadingCorrector(ShadingCorrector(),[1,n_reference]);
% bgw_data.turnOnRemoveDeadPixels();

%declare array to store mean in mean_array and variance in var_array
mean_array = zeros(n_reference*bgw_data.area,1);
var_array = zeros(n_reference*bgw_data.area,1);

%for each reference scan
for i_ref = 1:n_reference
    
    %get the stack of images from this scan
    image_array = bgw_data.reference_scan_array(i_ref).loadImageStack();
    %get the mean and variance
    mean_image = mean(image_array,3);
    var_image = var(image_array,[],3);
    
    %append the mean and variance to mean_array and var_array
    mean_array( ((i_ref-1)*bgw_data.area+1) : (i_ref*bgw_data.area) ) = reshape(mean_image,[],1);
    var_array( ((i_ref-1)*bgw_data.area+1) : (i_ref*bgw_data.area) ) = reshape(var_image,[],1);
end

%plot histogram
figure;
hist3Heatmap(mean_array,var_array,[100,100],true);
colorbar;
xlabel('mean (arb. unit)');
ylabel('variance  (arb. unit^2)');