%MIT License
%Copyright (c) 2019 Sherman Lo

%SCRIPT: COMPOUND POISSON HISTOGRAM AND DENSITY EVALUATION
%Simulate compound Poisson and plot the histogram along with the density
%The density can be evaluated exactly, using Normal approximation or saddlepoint approximation

close all;
clearvars;

%random seed
rng(uint32(353759542),'twister');

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

%for each parameter set
for iParameter = 1:nParameter
  
  %get parameters
  lambda = parameterArray(iParameter,1);
  alpha = parameterArray(iParameter,2);
  beta = parameterArray(iParameter,3);
  
  %simulated compound poisson
  X = CompoundPoisson.simulate(n,lambda,alpha,beta);
  
  %for each density evaluation method
  for iModel = 1:nModel
    
    %instantise a compound poisson object
    compoundPoisson = modelArray{iModel};
    
    %set the parameters of the compound poisson object
    %put the simulated data into the object
    compoundPoisson.setParameters(lambda,alpha,beta);
    compoundPoisson.addData(X);
    
    %plot the histogram
    plotHistogram(compoundPoisson, figureLocation);
    %plot qq plot
    plotQq(compoundPoisson, figureLocation);
  end
end


%NESTED FUNCTION: PLOT HISTOGRAM
%Plot histogram of the simulated compound Poisson
%Also plots frequency density with the normcdf([-1,1]) confidence intervals
%Plots zero frequency in a separate graph, the yLim is attempted to be on the same scale as the
    %frequency density plot
%Note to developers: requires constant bin width so that the confidence intervals can be
    %calculated.
    %The confidence intervals is calculated using N_bin ~ Poisson(pdf * N * binWidth) / binWidth.
    %This function can be extended for variable bin widths
%PARAMETERS:
  %compoundPoisson: stores the simulated compound Poisson and evaluates the density
  %isPlotZero: boolean, to plot the zero count or not
  %figureLocation: location to the save figure
function plotHistogram(compoundPoisson, figureLocation)

  %number of points when plotting densities
  nLinspace = 10000;
  %get data from compound poisson object
  X = compoundPoisson.X;
  n = compoundPoisson.n;
  
  %get an array of xs, equally spaced out across the range of the observed simulated data
  xPlot = linspace(min(X),max(X),nLinspace);
  %if the minimum of the data is 0, remove it
  if xPlot(1)==0
    xPlot(1) = [];
  end
  %get the pdf and mass at 0
  pdfPlot = compoundPoisson.getPdf(xPlot);
  if compoundPoisson.isSupportZeroMass
    p0 = compoundPoisson.getPdf(0);
  end
  
  %if statement for plotting histogram for zeros and positive numbers separately
  if (compoundPoisson.isSupportZeroMass && any(X==0))
    
    n0Obv = sum(X==0); %number of observed zeros
    n0Exp = n*p0; %expected number of zeros
    
    fig = LatexFigure.main();
    
    %bar chart plot the zero count
    subplot(1,2,1);
    bar(1, n0Obv);
    hold on;
    bar(2, n0Exp);
    %get error bars for the zero count, plot it as an horizontal line
    errorDown = poissinv(normcdf(-1),n0Exp);
    errorUp = poissinv(normcdf(1),n0Exp);
    errorXPlot = [0,3];
    plot(errorXPlot,[errorUp,errorUp],'r:');
    plot(errorXPlot,[errorDown,errorDown],'r:');
    %other graph properties being set
    ylabel('frequency at zero');
    xticks([1,2]);
    xticklabels({'obv.','expec.'});
    
    %histogram and density plot
    subplot(1,2,2);
    hist = Histogram(X(X~=0)); %histogram non-zero data
    hist.plot();
    
    %set the max of yLim to the max frequency in a bin observed 
    maxFreq = max([n0Obv,n0Exp,hist.n]) * 1.1;
    fig.Children(2).YLim(2) = maxFreq;
    fig.Children(1).YLim(2) = maxFreq / hist.binWidth(1); %assume all bins are the same width
  
  %if statement for not plotting the zero mass
  else
    %plot a regular histogram
    fig = LatexFigure.sub();
    hist = Histogram(X);
    hist.plot();
    ylim([0,max(hist.freqDensity)*1.1]);
  end
  
  %plot density with the error bars
  hold on;
  plot(xPlot,n*pdfPlot,'r');
  plot(xPlot,poissinv(normcdf(-1),n*pdfPlot*hist.binWidth(1))/hist.binWidth(1),'r:');
  plot(xPlot,poissinv(normcdf(1),n*pdfPlot*hist.binWidth(1))/hist.binWidth(1),'r:');
  xlabel('support');
  ylabel('frequency density');
  saveas(fig, ...
      fullfile(figureLocation, strcat(mfilename,'_',compoundPoisson.toString(),'.eps')),'epsc');
  
end

%NESTED FUNCTION: PLOT QQ PLOT
%Plots qq plot, simulated quantiles (p) vs theoretical quantiles (p) for an array of p
%PARAMETERS:
  %compoundPoisson: stores the simulated compound Poisson and evaluates the inverse cdf
  %figureLocation: location to save the figure
function plotQq(compoundPoisson, figureLocation)

  %number of trapeziums to be used, the trapezium rule is used for inverse cdf
  nTrapezium = 10000;
  %get data from compound poisson object
  X = compoundPoisson.X;
  n = compoundPoisson.n;

  fig = LatexFigure.sub();

  %get array of percentages
  p = ((1:n)'-0.5)/n;

  %get the inverse cdf for the array of percentages
  xTheoretical = compoundPoisson.getInvCdf(p, min(X), max(X), nTrapezium);

  %plot the simulated quantiles vs the exact quantiles
  scatter(xTheoretical,sort(X),'x');
  hold on;
  %plot straight line
  plot(xTheoretical,xTheoretical);
  xlabel('theoretical quantiles');
  ylabel('simulation quantiles');
  xlim([xTheoretical(1),xTheoretical(end)]);
  saveas(fig, ...
      fullfile(figureLocation, strcat(mfilename,'_qq_',compoundPoisson.toString(),'.eps')), 'epsc');

end