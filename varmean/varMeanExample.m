%MIT License
%Copyright (c) 2019 Sherman Lo

%VARIANCE MEAN EXAMPLE
%Plots the variance-mean frequency density heatmap
%Plots the GLM fits

clearvars;
close all;

plotExample(AbsNoFilterDeg30, {[0,0,0;0,1,0], [0,0,0;1,0,0]}, {'identity', 'reciprocal'});
plotExample(AbsFilterDeg30, {[0,0,0;1,0,0], [0,0,0;1,0,0]}, {'identity', 'reciprocal'});
plotExample(AbsNoFilterDeg120, {[0,0,0;0,1,0], [0,0,0;1,0,0]}, {'identity', 'reciprocal'});
plotExample(AbsFilterDeg120, {[0,0,0;1,0,0], [0,0,0;1,0,0]}, {'identity', 'reciprocal'});

%NESTED FUNCTION: PLOT EXAMPLE
%PARAMETERS:
  %scan: scan to get the var-mean data from
  %termMatrixArray: cell array of term matrices
    %element 1: x^{-1}
    %element 2: x
    %element 3: not used
  %linkArray: cell array of link functions
function plotExample(scan, termMatrixArray, linkArray)
  
  %add shading correction
  scan.addShadingCorrectorLinear();

  %get the grey values
  greyValueArray = getGreyValue(scan);
  %get var-mean data
  xOriginal = mean(greyValueArray,2);
  %use x^-1 and x features, they will be multiplied to get higher order features
  X = xOriginal.^([-1,1]);
  yOriginal = var(greyValueArray,[],2); %get the variance of the greyvalues
  
  %normalise the data
  xCentre = mean(X,1);
  xScale = std(X,[],1);
  X = (X-xCentre)./xScale; %noramlise
  yStd = std(yOriginal);
  y = yOriginal/yStd; %noramlise
  
  %for each model
  for i = 1:numel(termMatrixArray)
    
    %fit model
    termMatrix = termMatrixArray{i};
    link = linkArray{i};
    model = fitglm(X, y, termMatrix, 'Distribution', 'gamma', 'Link', link);
    
    %get shape parameter
    alpha = (scan.nSample - 1) / 2;
    
    %get the x-y values to plot the fit
    xPlot = linspace(min(xOriginal), max(xOriginal), 1000)';
    XPlot = xPlot.^([-1,1]);
    XPlot = (XPlot-xCentre)./xScale;
    yPlot = model.predict(XPlot) * yStd;
    
    %get the 1 sigma error bars
    upError = gaminv(0.975, alpha, yPlot/alpha);
    downError = gaminv(0.025, alpha, yPlot/alpha);
    
    %plot the heat map
    hist3Heatmap = Hist3Heatmap();
    fig = LatexFigure.sub();
    hist3Heatmap.plot(xOriginal, yOriginal);
    hold on;
    plot(xPlot, yPlot, 'r');
    plot(xPlot, upError, 'r--');
    plot(xPlot, downError, 'r--');
    xlabel('mean grey value (ADU)');
    ylabel('variance grey value (ADU²)');
    
    %save the figure
    terms = reshape(num2str(termMatrix'),1,[]);
    terms(terms==' ') = [];
    saveas(fig, fullfile('reports','figures','varmean', ...
        strcat(mfilename,'_',class(scan),'_',link,terms,'.eps')),'epsc');
    
  end
  
end