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
for alpha_i = 1:30
    for lambda_i = 1:30
        lnL(lambda_i,alpha_i) = model.lnL([nu_plot(lambda_i,alpha_i),80,lambda_plot(lambda_i,alpha_i)],X);
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

%SLICE CONTOUR PLOT
rng(116744603);
n = 100;
n_bin = 10;
nu = 500;
alpha = 80;
lambda = 2;
X = model.plotSimulation(n,nu,alpha,lambda,n_bin);
nu_lim = [200,1000];
alpha_lim = [40,120];
lambda_lim = [0.1,4];
n_grid = 30;
alpha_grid = 4;
[nu_plot,lambda_plot,alpha_plot] = meshgrid(linspace(nu_lim(1),nu_lim(2),n_grid),linspace(lambda_lim(1),lambda_lim(2),n_grid),linspace(alpha_lim(1),alpha_lim(2),alpha_grid));
lnL = zeros(n_grid,n_grid,alpha_grid);
for alpha_i = 1:alpha_grid
    for lambda_i = 1:n_grid
        for nu_i = 1:n_grid
            lnL(nu_i,lambda_i,alpha_i) = model.lnL([nu_plot(nu_i,lambda_i,alpha_i),alpha_plot(nu_i,lambda_i,alpha_i),lambda_plot(nu_i,lambda_i,alpha_i)],X);
        end
    end
end
figure;
contourslice(nu_plot,lambda_plot,alpha_plot,lnL,0,[],linspace(alpha_lim(1),alpha_lim(2),alpha_grid),50);
xlim(nu_lim);
ylim(lambda_lim);
zlim(alpha_lim);
xlabel('Poisson rate');
ylabel('Gamma rate');
zlabel('Gamma shape');

quiver_grid = 10;
[nu_plot,lambda_plot,alpha_plot] = meshgrid(linspace(nu_lim(1),nu_lim(2),quiver_grid),linspace(lambda_lim(1),lambda_lim(2),quiver_grid),linspace(alpha_lim(1),alpha_lim(2),alpha_grid));
grad_1 = zeros(quiver_grid,quiver_grid,alpha_grid);
grad_2 = zeros(quiver_grid,quiver_grid,alpha_grid);
grad_3 = zeros(quiver_grid,quiver_grid,alpha_grid);
grad_array = zeros(3,n);
for alpha_i = 1:alpha_grid
    for lambda_i = 1:quiver_grid
        for nu_i = 1:quiver_grid
            for n_i = 1:n
                grad_array(:,n_i) = model.gradient([nu_plot(nu_i,lambda_i,alpha_i),alpha_plot(nu_i,lambda_i,alpha_i),lambda_plot(nu_i,lambda_i,alpha_i)],X(n_i));
            end
            grad = sum(grad_array,2);
            grad_1(nu_i,lambda_i,alpha_i) = grad(1);
            grad_2(nu_i,lambda_i,alpha_i) = grad(2);
            grad_3(nu_i,lambda_i,alpha_i) = grad(3);
        end
    end
end
hold on;
d_nu = (nu_lim(2)-nu_lim(1))/(quiver_grid-1);
d_alpha = (alpha_lim(2)-alpha_lim(1))/(alpha_grid-1);
d_lambda = (lambda_lim(2)-lambda_lim(1))/(quiver_grid-1);
quiver3(nu_plot,lambda_plot,alpha_plot,-d_nu*grad_1,-d_lambda*grad_3,-d_alpha*grad_2,1,'Marker','o','MarkerSize',2,'AutoScale','off','ShowArrowHead','off');

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