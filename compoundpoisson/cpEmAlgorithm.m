%SCRIPT: COMPOUND POISSON EM ALGORITHM
%Fit the compound Poisson onto simulated data using the EM algorithm
%Plots, at each step of EM, the:
  %log likelihood (marginal, not joint)
  %lambda estimate
  %alpha estimate
  %beta estimate

close all;
clearvars;

%random seed
rng(uint32(2225638568),'twister');

n = 1000; %simulation sample size

%array of different ways to evaluate compound Poisson density
modelArray = cell(3,1);
modelArray{1} = CompoundPoisson();
modelArray{2} = CompoundPoissonNorm();
modelArray{3} = CompoundPoissonSaddle();
nModel = numel(modelArray);

%directory for the figures to be saved
figureLocation = fullfile('reports','figures','compoundpoisson');

%array of parameters to be investigated
  %dim 1: for each set
  %dim 2: lambda, alpha, beta
parameterArray = [
  1,1,1;
  10,1,1;
  1,100,1;
  100,100,1
  ];
%array to plot zero count for each of the parameters
isPlotZeroArray = [true, false, true, false];
nParameter = numel(parameterArray(:,1)); %number of parameters sets to be considered

nRepeat = 10; %number of times to repeat the experiment
nStep = 10; %number of EM steps
%for each parameter set
for iParameter = 1:nParameter
  %get parameters
  lambda = parameterArray(iParameter,1);
  alpha = parameterArray(iParameter,2);
  beta = parameterArray(iParameter,3);
  %plot the lnL, and parameters at each step of EM
  emAlgorithm(n, lambda, alpha, beta, nRepeat, nStep, figureLocation);
end

%NESTED FUNCTION: EM ALGORITHM
%Plots the log likelihood and parameters at each step of EM
%The starting points for the EM algorithm is at the true value
%PARAMETERS:
  %nSimulation: number of data points in the simulation
  %lambda: poisson parameter
  %alpha: gamma shape parameter
  %beta: gamma rate parameter
  %nRepeat: number of times to repeat the experiment
  %nStep: number of EM steps
  %figureLocation: directory location to save figures
function emAlgorithm(nSimulation, lambda, alpha, beta, nRepeat, nStep, figureLocation)

  %declare array of lnL, lambda, alpha and beta for each step of EM and each repeat
    %dim 1: for each step of EM
    %dim 2: for each repeat of the experiment
  lnLArray = zeros(nStep+1, nRepeat);
  lambdaArray = zeros(nStep+1, nRepeat);
  alphaArray = zeros(nStep+1, nRepeat);
  betaArray = zeros(nStep+1, nRepeat);

  %instantise a compound Poisson with the true parameters
  compoundPoissonTrue = CompoundPoisson();
  compoundPoissonTrue.setN(nSimulation);
  compoundPoissonTrue.setParameters(lambda, alpha, beta);

  %get the standard error of the estimators of the 3 parameters using the Fisher's information
      %matrix
  stdArray = sqrt(diag(inv(compoundPoissonTrue.getFisherInformation())));

  %for n_repeat times
  for iRepeat = 1:nRepeat

    %simulate n compound poisson varibales
    X = CompoundPoisson.simulate(nSimulation,lambda,alpha,beta);

    %set up a compound poisson random variable
    compoundPoisson = CompoundPoisson();
    compoundPoisson.setParameters(lambda,alpha,beta);
    compoundPoisson.addData(X);
    compoundPoisson.initaliseEM();

    %save the log likelihood, lambda, alpha and beta before EM
    lnLArray(1, iRepeat) = compoundPoisson.getMarginallnL();
    lambdaArray(1, iRepeat) = compoundPoisson.lambda;
    alphaArray(1, iRepeat) = compoundPoisson.alpha;
    betaArray(1, iRepeat) = compoundPoisson.beta;

    %for n_step times
    for iStep = 1:nStep
      %take a E and M step
      compoundPoisson.EStep();
      compoundPoisson.MStep();
      %save the log likelihood, lambda, alpha and beta before EM
      lnLArray(iStep+1, iRepeat) = compoundPoisson.getMarginallnL();
      lambdaArray(iStep+1, iRepeat) = compoundPoisson.lambda;
      alphaArray(iStep+1, iRepeat) = compoundPoisson.alpha;
      betaArray(iStep+1, iRepeat) = compoundPoisson.beta;
    end
  end

  %plot the log likelihood
  fig = LatexFigure.sub();
  plot(0:nStep, lnLArray, 'b');
  xlabel('number of EM steps');
  ylabel('lnL');
  xlim([0,nStep]);
  saveas(fig, ...
      fullfile(figureLocation, strcat(mfilename,'_',compoundPoissonTrue.toString(),...
      '_lnL.eps')),'epsc');

  %plot lambda
  fig = LatexFigure.sub();
  plot(0:nStep, lambdaArray, 'b');
  hold on;
  plot([0,nStep], [lambda-stdArray(1),lambda-stdArray(1)], 'k--');
  plot([0,nStep], [lambda+stdArray(1),lambda+stdArray(1)], 'k--');
  xlabel('number of EM steps');
  ylabel('\lambda');
  xlim([0,nStep]);
  saveas(fig, ...
      fullfile(figureLocation, strcat(mfilename,'_',compoundPoissonTrue.toString(),...
      '_lambda.eps')),'epsc');

  %plot alpha
  fig = LatexFigure.sub();
  plot(0:nStep, alphaArray, 'b');
  hold on;
  plot([0,nStep], [alpha-stdArray(2),alpha-stdArray(2)], 'k--');
  plot([0,nStep], [alpha+stdArray(2),alpha+stdArray(2)], 'k--');
  xlabel('number of EM steps');
  ylabel('\alpha');
  xlim([0,nStep]);
  saveas(fig, ...
      fullfile(figureLocation, strcat(mfilename,'_',compoundPoissonTrue.toString(),...
      '_alpha.eps')),'epsc');

  %plot beta
  fig = LatexFigure.sub();
  plot(0:nStep, betaArray, 'b');
  hold on;
  plot([0,nStep], [beta-stdArray(3),beta-stdArray(3)], 'k--');
  plot([0,nStep], [beta+stdArray(3),beta+stdArray(3)], 'k--');
  xlabel('number of EM steps');
  ylabel('\beta');
  xlim([0,nStep]);
  saveas(fig, ...
      fullfile(figureLocation, strcat(mfilename,'_',compoundPoissonTrue.toString(),...
      '_beta.eps')),'epsc');

end
