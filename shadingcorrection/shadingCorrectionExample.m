%SHADING CORRECTION EXAMPLE
%Plots the image with no shading correction, with bw and linear shading correction
%Plots the gradient image for each shading correction
%Plots the interpolation used in shading correction, that is:
  %y axis: within image mean
  %x axis: grey value
  %for a given pixel varying the current used to obtained the calibration images
  %plots best line of fit
  %this is repeated using a different colour and a differen pixel

clearvars;
close all;

seed = uint32(2146127166);
rng = RandStream('mt19937ar', 'Seed', seed); %random number generator
%rng used for selecting random pixels for the interpolation graph
%rng used for selecting random replication for training shading correction, another for the
    %resulting shading correction

%instantise scans with different shading corrections
scanArray(1) = AbsNoFilterDeg30(); %no shading correction
scanArray(2) = AbsNoFilterDeg30(); %bw shading correction
scanArray(3) = AbsNoFilterDeg30(); %linear shading correction
%use random replication for training the shading correction, and another for the resulting shading
    %correction
calibrationIndex = zeros(scanArray(1).calibrationScanArray(1).nSample, scanArray(1).whiteIndex);
for i = 1:scanArray(1).whiteIndex
  calibrationIndex(:,i) = rng.randperm(scanArray(1).calibrationScanArray(1).nSample)';
end
scanArray(2).addShadingCorrectorBw([calibrationIndex(1,1), calibrationIndex(1,end)]);
scanArray(3).addShadingCorrectorLinear(1:scanArray(3).whiteIndex, calibrationIndex(1,:));

%set the cLim for the gradient b
climB = ones(2,1);
%use the lowest and highest b found
for i = 1:numel(scanArray)
  scan = scanArray(i);
  if (scan.wantShadingCorrection)
    climB(1) = min([climB(1), min(min(scanArray(i).shadingCorrector.bArray))]);
    climB(2) = max([climB(2), max(max(scanArray(i).shadingCorrector.bArray))]);
  end
end

%use the min and max of the no shading correction image for the cLim
clim = [];
%for each shading correction
for i = 1:numel(scanArray)
  
  scan = scanArray(i); %get the scan, shading correction already applied
  
  %plot the scan
  fig = LatexFigure.sub();
  imagesc = Imagesc(scan.loadImage(1));
  if (~isempty(clim))
    imagesc.setCLim(clim);
  end
  imagesc.plot();
  clim = imagesc.clim; %save the clim
  saveas(fig,fullfile('reports','figures','data', ...
      strcat(mfilename,'_image_',scan.getShadingCorrectionStatus(),'.eps')),'epsc');
  
  %get the black and white image
  black = scan.calibrationScanArray(1).loadImage(calibrationIndex(2,1));
  white = scan.calibrationScanArray(scan.whiteIndex).loadImage(calibrationIndex(2,end));
  
  %plot the black image
  fig = LatexFigure.sub();
  imagesc = Imagesc(black);
  imagesc.plot();
  saveas(fig,fullfile('reports','figures','data', ...
      strcat(mfilename,'_black_',scan.getShadingCorrectionStatus(),'.eps')),'epsc');
  
  %plot the white image
  fig = LatexFigure.sub();
  imagesc = Imagesc(white);
  imagesc.setCLim(clim);
  imagesc.plot();
  saveas(fig,fullfile('reports','figures','data', ...
      strcat(mfilename,'_white_',scan.getShadingCorrectionStatus(),'.eps')),'epsc');
  
  %plot the fourier transform of the black and white image
  fig = LatexFigure.sub();
  imagesc = Imagesc(fft(black));
  imagesc.plot();
  saveas(fig,fullfile('reports','figures','data', ...
      strcat(mfilename,'_blackFft_',scan.getShadingCorrectionStatus(),'.eps')),'epsc');
  
  fig = LatexFigure.sub();
  imagesc = Imagesc(fft(white));
  imagesc.plot();
  saveas(fig,fullfile('reports','figures','data', ...
      strcat(mfilename,'_whiteFft_',scan.getShadingCorrectionStatus(),'.eps')),'epsc');
    
  %plot the greyvalue vs power
  greyvalueArray = zeros(scan.area, scan.whiteIndex);
  for iPower = 1:scan.whiteIndex
    greyvalueArray(:,iPower) = ...
        reshape(scan.calibrationScanArray(iPower).loadImage(calibrationIndex(2,iPower)), [], 1);
  end
  fig = LatexFigure.sub();
  boxplotPower = Boxplots(greyvalueArray);
  boxplotPower.setPosition(scan.getPowerArray());
  boxplotPower.setWantOutlier(false);
  boxplotPower.plot();
  ylabel('grey value (ADU)');
  xlabel('power (W)');
  saveas(fig,fullfile('reports','figures','data', ...
      strcat(mfilename,'_power_',scan.getShadingCorrectionStatus(),'.eps')),'epsc');
  
  %plot the gradient if this has shading correction
  if (scan.wantShadingCorrection)
    fig = LatexFigure.sub();
    imagesc = Imagesc(scan.shadingCorrector.bArray);
    imagesc.setCLim(climB);
    imagesc.plot();
    saveas(fig,fullfile('reports','figures','data', ...
        strcat(mfilename,'_gradient_',scan.getShadingCorrectionStatus(),'.eps')),'epsc');
  end
  
  
end

%plot within calibration image mean vs greyvalue for each calibration image for a given pixel
%also plot best line of fit
%pixel nPlot random pixels
nPlot = 3;
scan = scanArray(1); %get the no shading correction images
yArray = zeros(scan.whiteIndex, 1); %array of within calibration image mean
%array of greyvalue, dim 1: for each power, dim 2: for each nPlot
xArray = zeros(scan.whiteIndex, nPlot);
%get random index
sampleIndex = rng.randi([1,scan.area], 1, nPlot);
%fill in yArray and xArray
for i = 1:scan.whiteIndex
  yArray(i) = mean(reshape(scan.calibrationScanArray(i).loadImage(1),[],1));
  calibrationImage = scan.calibrationScanArray(i).loadImage(1);
  xArray(i,:) = calibrationImage(sampleIndex);  
end

%scatter plot the within image mean vs greyvalue
fig = LatexFigure.sub();
hold on;
ax = gca;
for i = 1:nPlot
  scatter(xArray(:,i), yArray, 'x', 'MarkerEdgeColor', ax.ColorOrder(i,:));
end
ax = gca;
%plot the best line of fit
xPlot = ax.XLim';
yPlot = zeros(2, nPlot);
shadingCorrector = scanArray(3).shadingCorrector; %use linear shading corrector
yPlot(1,:) = shadingCorrector.bArray(sampleIndex) ...
    .* (xPlot(1) - shadingCorrector.betweenReferenceMean(sampleIndex)) ...
    + shadingCorrector.globalMean;
yPlot(2,:) = shadingCorrector.bArray(sampleIndex) ...
    .* (xPlot(2) - shadingCorrector.betweenReferenceMean(sampleIndex)) ...
    + shadingCorrector.globalMean;
for i = 1:nPlot
  plot(xPlot, yPlot(:,i), 'Color',  ax.ColorOrder(i,:));
end
%force the ylim to start at zero
ax.YLim(1) = 0;
xlabel('grey value (ADU)');
ylabel('within image mean (ADU)');
legend('pixel 1', 'pixel 2', 'pixel 3', 'Location', 'northwest');
saveas(fig,fullfile('reports','figures','data', strcat(mfilename,'_interpolation.eps')),'epsc');

%use imageJ for FFT
function imageFft = fft(image)
  imagePlus = ij.ImagePlus('', ij.process.FloatProcessor(image));
  transform = ij.plugin.FFT.forward(imagePlus);
  imageFft = reshape(transform.getProcessor.getPixels(), ...
      transform.getWidth(), transform.getHeight())';
  imageFft = double(bitand(int16(imageFft),int16(255)));
end