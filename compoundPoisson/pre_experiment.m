clc;
clearvars;
close all;

%set variables
model = CompoundPoisson_msle(1); %compound poisson model with time exposure
n = 1E4; %number of data in sample
n_bin = 50; %number of bins for the histogram

%EXPERIMENT 1: PLOT HISTOGRAM OF SIMULATED DATA AND SADDLE DENSITY

%parameter set 1
rng(51089120);
model.plotSimulation(n,1,1,1,n_bin);
xlim([0,5]);
ylim([0,5]);

%parameter set 2
rng(87091791);
model.plotSimulation(n,3,400,100,n_bin);

%parameter set 3
rng(116744603);
X = model.plotSimulation(n,500,80,2,n_bin);
%also surf and contour plot the minus log likelihood
[nu_plot,lambda_plot] = meshgrid(linspace(1,1000,30),linspace(0.1,5,30));
lnL = zeros(30,30);
for i = 1:30
    for j = 1:30
        lnL(j,i) = model.lnL([nu_plot(j,i),80,lambda_plot(j,i)],X);
    end
end
figure;
surf(nu_plot,lambda_plot,lnL);
xlabel('nu');
ylabel('lambda');
zlabel('-lnL');
figure;
contour(nu_plot,lambda_plot,lnL,200);
xlabel('nu');
ylabel('lambda');
zlabel('-lnL');

%EXPERIMENT 2
%Plot emperical sampling distribution when estimating the parameters

%number of estimators
n_repeat = 1000;
%number of bins for the histogram
n_bin = 10;
%sample size of the data
n = 100;

%parameter set 1
rng(172751188);
model.plotSamplingDistribution(n_repeat,n,1,1,1,n_bin);

%parameter set 2
rng(138739981);
model.plotSamplingDistribution(n_repeat,n,3,400,100,n_bin);

%parameter set 3
rng(81190838);
model.plotSamplingDistribution(n_repeat,n,500,80,2,n_bin);