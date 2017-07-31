%VARIANCE MEAN SCRIPT
%Plots a histogram heat map of variance-mean pair
%of each pixel from each reference scan
%Plots mean pixel vs power
%Plots variance pixel vs power

clc;
clearvars;
close all;

%get the data
bgw_data = AbsBlock_Sep16_30deg();
area = bgw_data.area;
n_reference = numel(bgw_data.reference_scan_array);

power_array = zeros(1,n_reference);
for i = 1:n_reference
    power_array(i) = bgw_data.reference_scan_array(i).power;
end

%apply shading correction

bgw_data.addShadingCorrector(ShadingCorrector(),1:n_reference,ones(1,n_reference));
bgw_data.turnOnRemoveDeadPixels();

% bgw_data.addShadingCorrector(ShadingCorrector(),[1,n_reference],[1,1]);
% bgw_data.turnOnRemoveDeadPixels();

%declare array to store mean in mean_array and variance in var_array
mean_array = zeros(bgw_data.area,n_reference);
var_array = zeros(bgw_data.area,n_reference);
var_b_array = zeros(n_reference,1);
var_w_array = zeros(n_reference,1);

%for each reference scan
for i_ref = 1:n_reference
    
    reference_scan = bgw_data.reference_scan_array(i_ref);
    
    %get the stack of images from this scan
    image_array = reference_scan.loadImageStack(2:reference_scan.n_sample);
    %get the mean and variance
    mean_image = mean(image_array,3);
    var_image = var(image_array,[],3);
    
    %append the mean and variance to mean_array and var_array
    mean_array(:, i_ref) = reshape(mean_image,[],1);
    var_array(:, i_ref) = reshape(var_image,[],1);
    [var_b_array(i_ref),var_w_array(i_ref)] = var_between_within(image_array);
end

%work out the errors for var_b and var_w
var_b_error = zeros(2,n_reference);
var_w_error = zeros(2,n_reference);

sig_level = 0.5;
for i = 1:n_reference
    
    dof = area - 1;
    var_b_error(:,i) = abs(dof*var_b_array(i)./chi2inv([1-(sig_level/2);sig_level/2],dof) - var_b_array(i));
    
    dof = area*bgw_data.reference_scan_array(i_ref).n_sample - area;
    var_w_error(:,i) = abs(dof*var_w_array(i)./chi2inv([1-(sig_level/2);sig_level/2],dof) - var_w_array(i));
end

%plot histogram
figure;
hist3Heatmap(reshape(mean_array,[],1),reshape(var_array,[],1),[100,100],true);
colorbar;
xlabel('mean (arb. unit)');
ylabel('variance  (arb. unit^2)');

figure;
plot(power_array,mean(mean_array),'LineStyle','none');
ax = gca;
x_tick = ax.XTick;
x_tick_label = ax.XTickLabel;
boxplot(ax,mean_array,'position', power_array,'boxstyle','filled','medianstyle','target','outliersize',4,'symbol','o');
ax = gca;
ax.XTick = x_tick;
ax.XTickLabel = x_tick_label;
xlabel('Power (W)');
ylabel('Pixel mean (arb. unit)');

figure;
boxplot(var_array,'position', power_array,'boxstyle','filled','medianstyle','target','outliersize',4,'symbol','o');
ax = gca;
ax.XTick = x_tick;
ax.XTickLabel = x_tick_label;
xlabel('Power (W)');
ylabel('Pixel variance (arb. unit^2)');
