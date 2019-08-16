%MIT License
%Copyright (c) 2019 Sherman Lo

%SCRIPT: PLOTTING LOG LIKELIHOOD
%Plots the log likelihood as a function of alpha and beta for fixed lambda

clearvars;
close all;

%random seed
rng(uint32(3270361853),'twister');

n = 100;

lambda = 1;
alpha = 1;
beta = 1;
[X,~] = CompoundPoisson.simulate(n,lambda,alpha,beta);
compoundPoisson = CompoundPoisson();
compoundPoisson.addData(X);
compoundPoisson.setParameters(lambda,alpha,beta);
cpPlotlnL(compoundPoisson, [0.5,1.5], [0.5,1.5], 20);

lambda = 1;
alpha = 100;
beta = 1;
[X,~] = CompoundPoisson.simulate(n,lambda,alpha,beta);
compoundPoisson = CompoundPoisson();
compoundPoisson.addData(X);
compoundPoisson.setParameters(lambda,alpha,beta);
cpPlotlnL(compoundPoisson, [50,150], [0.5,1.5], 20);

lambda = 10;
alpha = 1;
beta = 1;
[X,~] = CompoundPoisson.simulate(n,lambda,alpha,beta);
compoundPoisson = CompoundPoisson();
compoundPoisson.addData(X);
compoundPoisson.setParameters(lambda,alpha,beta);
cpPlotlnL(compoundPoisson, [0.5,1.5], [0.5,1.5], 20);

lambda = 100;
alpha = 100;
beta = 1;
[X,~] = CompoundPoisson.simulate(n,lambda,alpha,beta);
compoundPoisson = CompoundPoisson();
compoundPoisson.addData(X);
compoundPoisson.setParameters(lambda,alpha,beta);
cpPlotlnL(compoundPoisson, [50,150], [0.5,1.5], 20);

%NESTED FUNCTION: COMPOUND POISSON PLOT LOG LIKELIHOOD
%Plots the log likelihood as a function of alpha and beta
%PARAMETERS:
  %compoundPoisson: compound poisson object with data and parameters
  %alpha: 2 vector, alpha values limit for plot
  %beta: 2 vector, beta values limit for plot
  %n: number of points in an axis
function cpPlotlnL(compoundPoisson, alpha, beta, n)

  %make a copy of the compound poisson
  compoundPoissonTrue = CompoundPoisson();
  compoundPoissonTrue.setParameters(...
      compoundPoisson.lambda, compoundPoisson.alpha, compoundPoisson.beta);

  %get a mesh grid of alphas and betas
  [alphaGrid, betaGrid] = meshgrid(linspace(alpha(1),alpha(2),n),linspace(beta(1),beta(2),n));
  %get a mesh grid of the log likelihoods
  lnLFullGrid = alphaGrid;
  %for each value in the grid
  for i = 1:numel(lnLFullGrid)
    %set the compound poisson to have that parameter
    compoundPoisson.setParameters(compoundPoisson.lambda,alphaGrid(i),betaGrid(i));
    %get the log likelihood
    lnLFullGrid(i) = compoundPoisson.getMarginallnL();
  end

  %surf plot the log likelihood
  fig = LatexFigure.sub();
  surf(alphaGrid, betaGrid, lnLFullGrid);
  xlabel('\alpha');
  ylabel('\beta');
  zlabel('log likelihood');
  view(-45,35.264);
  saveas(fig, fullfile(fullfile('reports','figures','compoundpoisson'), ...
      strcat(mfilename,'_',compoundPoissonTrue.toString(),'.eps')),'epsc');

end