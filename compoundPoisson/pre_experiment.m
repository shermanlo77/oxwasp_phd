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
[nu_plot,lambda_plot,alpha_plot] = meshgrid(linspace(200,1000,30),linspace(0.1,4,30),linspace(40,120,3));
lnL = zeros(30,30,3);
for i = 1:3
    for j = 1:30
        for k = 1:30 
            lnL(k,j,i) = model.lnL([nu_plot(k,j,i),alpha_plot(k,j,i),lambda_plot(k,j,i)],X);
        end
    end
end
figure;
contourslice(nu_plot,lambda_plot,alpha_plot,lnL,0,[],linspace(40,120,3),50);

[nu_plot,lambda_plot,alpha_plot] = meshgrid(linspace(200,1000,10),linspace(0.1,4,10),linspace(40,120,3));
grad_1 = zeros(10,10,3);
grad_2 = zeros(10,10,3);
grad_3 = zeros(10,10,3);
parfor i = 1:3
    for j = 1:10
        for k = 1:10 
            grad = model.gradient([nu_plot(k,j,i),alpha_plot(k,j,i),lambda_plot(k,j,i)],2E4);
            grad_1(k,j,i) = grad(1);
            grad_2(k,j,i) = grad(2);
            grad_3(k,j,i) = grad(3);
        end
    end
end
hold on;
quiver3(nu_plot,lambda_plot,alpha_plot,grad_1,grad_3,grad_2,0.1,'Marker','o','MarkerSize',2);

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