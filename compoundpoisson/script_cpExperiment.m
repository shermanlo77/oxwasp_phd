%SCRIPT FOR COMPOUND POISSON
%Plots histogram with density of simulated compound Poisson
%plots qq plot comparing histogram and density
%density evaluation using exact method, normal approx and saddlepoint approx

clc;
close all;
clearvars;

%random seed
rng(uint32(353759542),'twister');

n = 1000; %simulation sample size
n_density = 3; %number of density evaluation methods
%directory for the figures to be saved
figure_location = fullfile('reports','figures','compoundPoisson');

%array of parameters to be investigated
    %dim 1: for each set
    %dim 2: lambda, alpha, beta
parameter_array = [
    1,1,1;
    10,1,1;
    1,100,1;
    100,100,1
];
n_parameter = numel(parameter_array(:,1)); %number of parameter sets to be considered

%for each parameter set
for i_parameter = 1:n_parameter
    
    %get parameters
    lambda = parameter_array(i_parameter,1);
    alpha = parameter_array(i_parameter,2);
    beta = parameter_array(i_parameter,3);
    
    %simulated compound poisson
    X = CompoundPoisson.simulate(n,lambda,alpha,beta);
   
    %for each density evaluation method
    for i_density = 1:n_density
       
        %instantise a compound poisson object for a given i_density
        %different i_density will change how each compound poisson object evaluate the pdf
        switch i_density
            case 1
                compound_poisson = CompoundPoisson();
            case 2
                compound_poisson = CompoundPoisson_norm();
            case 3
                compound_poisson = CompoundPoisson_saddle();
        end
        
        %set the parameters of the compound poisson object
        %put the simulated data into the object
        compound_poisson.setParameters(lambda,alpha,beta);
        compound_poisson.addData(X);
        
        %assign file names for the histogram and qq plot
        hist_name = strcat(compound_poisson.name,'_hist',num2str(i_parameter),'.png');
        qq_name = strcat(compound_poisson.name,'_qq',num2str(i_parameter),'.png');
        
        %save histogram and qq plot
        cpQQDensity(compound_poisson,fullfile(figure_location,hist_name),fullfile(figure_location,qq_name));
       
   end
end

n_repeat = 10; %number of times to repeat the experiment
n_step = 10; %number of EM steps
%for each parameter set
for i_parameter = 1:n_parameter
    %get parameters
    lambda = parameter_array(i_parameter,1);
    alpha = parameter_array(i_parameter,2);
    beta = parameter_array(i_parameter,3);
    %plot the lnL, and parameters at each step of EM
    cpConvergence(n, lambda, alpha, beta, n_repeat, n_step,figure_location,strcat('convergence_',num2str(i_parameter)));
end