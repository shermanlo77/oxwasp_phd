clc;
clearvars;
close all;

%PLOTS the saddlepoint approximated log likelihood for the compound Poisson distribution
%lnL is a function of 3 parameters of the compound Poisson distribution:
    %nu: poisson rate (x axis)
    %alpha: gamma shape (z axis and sliced so that for each alpha, there is a contour plot)
    %lambda: gamma rate (y axis)

%instantiate saddle point object
model = CompoundPoisson_saddlePoint(1); %compound poisson model with time exposure

%set variables
rng(116744603); %set random seed
n = 1E4; %number of simulations
n_bin = 20; %number of bins for the histogram
nu = 500; %poisson rate
alpha = 80; %gamma shape
lambda = 2; %gamma rate

%get n simulations of the compound Poisson
X = model.plotSimulation(n,nu,alpha,lambda,n_bin);

%set axis limits of the log likelihood vs 3 compound Poisson parameters
nu_lim = [200,1000];
alpha_lim = [40,120];
lambda_lim = [0.1,4];
n_grid = 30; %number of increments for the meshgrid
alpha_grid = 4; %number of slices in the alpha axis
%meshgrid
[nu_plot,lambda_plot,alpha_plot] = meshgrid(linspace(nu_lim(1),nu_lim(2),n_grid),linspace(lambda_lim(1),lambda_lim(2),n_grid),linspace(alpha_lim(1),alpha_lim(2),alpha_grid));
%declare 3d array for log likelihood
lnL = zeros(n_grid,n_grid,alpha_grid);
%for each possible value for each compound Poisson parameter
for alpha_i = 1:alpha_grid
    for lambda_i = 1:n_grid
        for nu_i = 1:n_grid
            %work out the log likelihood and save it to the 3d array lnL
            lnL(nu_i,lambda_i,alpha_i) = model.lnL([nu_plot(nu_i,lambda_i,alpha_i),alpha_plot(nu_i,lambda_i,alpha_i),lambda_plot(nu_i,lambda_i,alpha_i)],X);
        end
    end
end

%contour slice plot the log likelihood
figure;
contourslice(nu_plot,lambda_plot,alpha_plot,lnL,0,[],linspace(alpha_lim(1),alpha_lim(2),alpha_grid),50);
xlim(nu_lim);
ylim(lambda_lim);
zlim(alpha_lim);
xlabel('Poisson rate');
ylabel('Gamma rate');
zlabel('Gamma shape');
view(-36,19);

saveas(gca,'reports/figures/compoundPoisson/log_likelihood.eps','epsc2');

%vector field plot for the gradient

% quiver_grid = 10;
% [nu_plot,lambda_plot,alpha_plot] = meshgrid(linspace(nu_lim(1),nu_lim(2),quiver_grid),linspace(lambda_lim(1),lambda_lim(2),quiver_grid),linspace(alpha_lim(1),alpha_lim(2),alpha_grid));
% grad_1 = zeros(quiver_grid,quiver_grid,alpha_grid);
% grad_2 = zeros(quiver_grid,quiver_grid,alpha_grid);
% grad_3 = zeros(quiver_grid,quiver_grid,alpha_grid);
% grad_array = zeros(3,n);
% for alpha_i = 1:alpha_grid
%     for lambda_i = 1:quiver_grid
%         for nu_i = 1:quiver_grid
%             for n_i = 1:n
%                 grad_array(:,n_i) = model.gradient([nu_plot(nu_i,lambda_i,alpha_i),alpha_plot(nu_i,lambda_i,alpha_i),lambda_plot(nu_i,lambda_i,alpha_i)],X(n_i));
%             end
%             grad = sum(grad_array,2);
%             grad_1(nu_i,lambda_i,alpha_i) = grad(1);
%             grad_2(nu_i,lambda_i,alpha_i) = grad(2);
%             grad_3(nu_i,lambda_i,alpha_i) = grad(3);
%         end
%     end
% end
% hold on;
% d_nu = (nu_lim(2)-nu_lim(1))/(quiver_grid-1);
% d_alpha = (alpha_lim(2)-alpha_lim(1))/(alpha_grid-1);
% d_lambda = (lambda_lim(2)-lambda_lim(1))/(quiver_grid-1);
% quiver3(nu_plot,lambda_plot,alpha_plot,-d_nu*grad_1,-d_lambda*grad_3,-d_alpha*grad_2,1,'Marker','o','MarkerSize',2,'AutoScale','off','ShowArrowHead','off');
