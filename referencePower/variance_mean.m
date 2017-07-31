%VARIANCE MEAN SCRIPT
%Plots a histogram heat map of variance-mean pair
%of each pixel from each reference scan

clc;
clearvars;
close all;

%get the data
bgw_data = AbsBlock_July16_30deg();
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
var_b_array = zeros(n_reference,1);
var_w_array = zeros(n_reference,1);

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
    [var_b_array(i_ref),var_w_array(i_ref)] = var_between_within(image_array);
end

%work out the errors for var_b and var_w
var_b_error = zeros(2,n_reference);
var_w_error = zeros(2,n_reference);

sig_level = 0.5;
for i = 1:n_reference
    
    dof = n_reference - 1;
    var_b_error(:,i) = abs(dof*var_b_array(i)./chi2inv([1-(sig_level/2);sig_level/2],dof) - var_b_array(i));
    
    dof = bgw_data.area*n_reference - n_reference;
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

figure;
errorbar(power_array,var_b_array,var_b_error(1,:),var_b_error(2,:),'LineStyle','none');
hold on;
errorbar(power_array,var_w_array,var_w_error(1,:),var_w_error(2,:),'LineStyle','none');
set(gca,'yscale','log')
xlabel('Power (W)');
legend('Between pixel','Within pixel');
ylabel('Variance (arb. unit^2)');
xlim([power_array(1)-1,power_array(end)+1]);
