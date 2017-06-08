% clc;
% clearvars;
% close all

bgw_data = BGW_140316();
% bgw_data.addShadingCorrector(ShadingCorrector(),[1,1,1]);
% bgw_data.turnOnSetExtremeToNan();
% bgw_data.turnOnRemoveDeadPixels();


power_array = [0, 1.7, 6.8];

mean_array = zeros(bgw_data.area,3);
variance_array = zeros(bgw_data.area,3);

mean_array(:,1) = reshape(mean(bgw_data.loadBlackStack(2:20),3),[],1);
mean_array(:,2) = reshape(mean(bgw_data.loadGreyStack(2:20),3),[],1);
mean_array(:,3) = reshape(mean(bgw_data.loadWhiteStack(2:20),3),[],1);

Y = reshape(mean_array,[],1);
X = ones(3*bgw_data.area,2);
X(1:bgw_data.area,2) = power_array(1);
X((bgw_data.area+1):(2*bgw_data.area),2) = power_array(2); 
X((2*bgw_data.area+1):(3*bgw_data.area),2) = power_array(3);
b_mean = X\Y;

x_plot_mean = [1,power_array(1);1,power_array(end)];
y_plot_mean = x_plot_mean'*b_mean;

figure;
boxplot(mean_array, power_array, 'positions', power_array);
hold on;
plot(x_plot_mean(:,2),y_plot_mean);
hold off;


variance_array(:,1) = reshape(var(bgw_data.loadBlackStack(2:20),[],3),[],1);
variance_array(:,2) = reshape(var(bgw_data.loadGreyStack(2:20),[],3),[],1);
variance_array(:,3) = reshape(var(bgw_data.loadWhiteStack(2:20),[],3),[],1);

Y = reshape(variance_array,[],1);
b_var = X\Y;

x_plot_var = [1,power_array(1);1,power_array(end)];
y_plot_var = x_plot_var'*b_var;

figure;
boxplot(variance_array, power_array, 'positions', power_array);
hold on;
plot(x_plot_var(:,2),y_plot_var);
hold off;