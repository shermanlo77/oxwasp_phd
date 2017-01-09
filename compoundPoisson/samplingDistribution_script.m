clc;
clearvars;
close all;

%Sampling distribution script
%Plot emperical sampling distribution when estimating the parameters

%set variables
model = CompoundPoisson_saddlePoint(1); %compound poisson model with time exposure

%number of estimators
n_repeat = 1000;
%number of bins for the histogram
n_bin = 10;
%sample size of the data
n = 100;

%parameter set 2
rng(138739981);
ax_array = model.plotSamplingDistribution(n_repeat,n,3,400,100,n_bin);
%saveas(ax_array{1},'reports/figures/compoundPoisson/sampling_dist_1_nu','epsc2');
%saveas(ax_array{2},'reports/figures/compoundPoisson/sampling_dist_1_alpha','epsc2');
%saveas(ax_array{3},'reports/figures/compoundPoisson/sampling_dist_1_lambda','epsc2');

%parameter set 3
rng(81190838);
ax_array = model.plotSamplingDistribution(n_repeat,n,500,80,2,n_bin);
%saveas(ax_array{1},'reports/figures/compoundPoisson/sampling_dist_2_nu','epsc2');
%saveas(ax_array{2},'reports/figures/compoundPoisson/sampling_dist_2_alpha','epsc2');
%saveas(ax_array{3},'reports/figures/compoundPoisson/sampling_dist_2_lambda','epsc2');