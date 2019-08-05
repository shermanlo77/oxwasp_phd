%SCRIPT: VAR MEAN RESIDUAL
%Plot the residual vs greyvalue and residual on the projections

close all;
clearvars;

directory = fullfile('reports','figures','varmean');

%list of datasets
scanArray = cell(1,1);
scanArray{1} = AbsNoFilterDeg30();
scanArray{2} = AbsNoFilterDeg120();
scanArray{3} = AbsFilterDeg30();
scanArray{4} = AbsFilterDeg120();
for iScan = 1:numel(scanArray)
  scanArray{iScan}.addShadingCorrectorLinear();
end
link = 'identity';
termMatrix = [0,0;1,0];

%for each data
for iScan = 1:numel(scanArray)
  
  scan = scanArray{iScan};
  %get the grey values
  greyValueArray = getGreyValue(scan);
  %get var-mean data
  xOriginal = mean(greyValueArray,2);
  X = xOriginal;
  y = var(greyValueArray,[],2); %get the variance of the greyvalues
  
  %normalise the data
  xCentre = mean(X,1);
  xScale = std(X,[],1);
  X = (X-xCentre)./xScale; %noramlise
  yStd = std(y);
  y = y/yStd; %noramlise
  
  %fit model
  model = fitglm(X, y, termMatrix, 'Distribution', 'gamma', 'Link', link);
  residual = (y - table2array(model.Fitted(:,1))) * yStd;
  
  %get shape parameter
  alpha = (scan.nSample - 1) / 2;

  %get confidence intervals for the residuals
  xPlot = linspace(min(xOriginal), max(xOriginal), 1000)';
  XPlot = (xPlot-xCentre)./xScale;
  yPlot = model.predict(XPlot) * yStd;
  upError = gaminv(normcdf(1), alpha, yPlot/alpha) - yPlot;
  downError = gaminv(normcdf(-1), alpha, yPlot/alpha) - yPlot;

  %plot the heat map of the residual vs grey value
  hist3Heatmap = Hist3Heatmap();
  fig = LatexFigure.sub();
  hist3Heatmap.plot(xOriginal, residual);
  hold on;
  plot(xPlot, upError, 'r--');
  plot(xPlot, downError, 'r--');
  xlabel('mean grey value (ADU)');
  ylabel('residual (ADU²)');
  saveas(fig, fullfile(directory,strcat(mfilename,class(scan),'_vsgreyvalue.eps')), 'epsc');
  
  %plot the residual on the segmentation
  residualSpatial = double(scan.getSegmentation());
  residualSpatial(residualSpatial==1) = residual;
  residualSpatial(residualSpatial==0) = nan;
  fig = LatexFigure.sub();
  imagesc = Imagesc(abs(residualSpatial));
  imagesc.setCLim([0,max(upError)]);
  if (isa(scan,'AbsNoFilterDeg120'))
    highlight = false(scan.height, scan.width);
    highlight(130:1860,220) = true;
    highlight(130:1860,620) = true;
    highlight(133,220:620) = true;
    highlight(1860,220:620) = true;
    imagesc.addPositivePixels(highlight);
    imagesc.setDilateSize(15);
  end
  imagesc.plot();
  saveas(fig, fullfile(directory,strcat(mfilename,class(scan),'_spatial.eps')), 'epsc');
  
end