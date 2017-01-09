%APPROXIMATION SCRIPT

%For 3 examples of parameters, simulate compound Poisson and plot histogram of the simulations.
%the density approximation is plotted on top of that 

clc;
clearvars;
close all;

%set variables
saddle_point = CompoundPoisson_saddlePoint(1); %compound poisson model with time exposure
normal_approx = CompoundPoisson_normal(1); %compound poisson model with time exposure
n = 1E4; %number of data in sample
n_bin = 20; %number of bins for the histogram

%PLOT HISTOGRAM OF SIMULATED DATA AND DENSITY APPROXIMATION


%PARAMETER SET 1, Poisson rate = 1, gamma shape = 1, gamma rate = 1

%normal approximation
rng(51089120);
normal_approx.plotSimulation(n,1,1,1,n_bin);
xlim([0,5]);
ylim([0,5]);
saveas(gca,'reports/figures/compoundPoisson/normal_1.eps','epsc2');

%saddle point approximation
rng(51089120);
saddle_point.plotSimulation(n,1,1,1,n_bin);
xlim([0,5]);
ylim([0,5]);
saveas(gca,'reports/figures/compoundPoisson/saddle_1.eps','epsc2');


%PARAMETER SET 2, Poisson rate = 3, gamma shape = 400, gamma rate = 100

%normal approximation
rng(87091791);
normal_approx.plotSimulation(n,3,400,100,n_bin);
saveas(gca,'reports/figures/compoundPoisson/normal_2.eps','epsc2');

%saddle point approximation
rng(87091791);
saddle_point.plotSimulation(n,3,400,100,n_bin);
saveas(gca,'reports/figures/compoundPoisson/saddle_2.eps','epsc2');

%PARAMETER SET 3, poisson rate = 500, gamma shape = 80, gamma rate = 2

%normal approximation
rng(116744603);
normal_approx.plotSimulation(n,500,80,2,n_bin);
saveas(gca,'reports/figures/compoundPoisson/normal_3.eps','epsc2');

%saddle point approximation
rng(116744603);
saddle_point.plotSimulation(n,500,80,2,n_bin);
saveas(gca,'reports/figures/compoundPoisson/saddle_3.eps','epsc2');
