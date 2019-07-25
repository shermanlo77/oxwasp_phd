%SCRIPT FOR PLOTTING THE LOG LIKELIHOOD
%Plots the log likelihood as a function of alpha and beta

clc;
clearvars;
close all;

lambda = 1;
alpha = 1;
beta = 1;
n = 100;
[X,~] = CompoundPoisson.simulate(n,lambda,alpha,beta);
compound_poisson = CompoundPoisson();
compound_poisson.addData(X);
compound_poisson.setParameters(lambda,alpha,beta);
cpPlotlnL(compound_poisson, [0.5,1.5], [0.5,1.5], 20);

lambda = 1;
alpha = 100;
beta = 1;
n = 100;
[X,~] = CompoundPoisson.simulate(n,lambda,alpha,beta);
compound_poisson = CompoundPoisson();
compound_poisson.addData(X);
compound_poisson.setParameters(lambda,alpha,beta);
cpPlotlnL(compound_poisson, [50,150], [0.5,1.5], 20);

lambda = 100;
alpha = 100;
beta = 1;
n = 100;
[X,~] = CompoundPoisson.simulate(n,lambda,alpha,beta);
compound_poisson = CompoundPoisson();
compound_poisson.addData(X);
compound_poisson.setParameters(lambda,alpha,beta);
cpPlotlnL(compound_poisson, [50,150], [0.5,1.5], 20);