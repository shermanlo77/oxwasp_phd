%MIT License
%Copyright (c) 2019 Sherman Lo

%PLOT SCANS
%Plot for each dataset, for each angle:
  %x-ray scan
  %x-ray simulation
  %calibration scans
  %calibration simulations
%calibration images only go up to the maximum power used in the x-ray acquisition

close all;
clearvars;

plotAndSaveScan([AbsNoFilterDeg30(), AbsNoFilterDeg120()]);
plotAndSaveScan([AbsFilterDeg30(), AbsFilterDeg120()]);
plotAndSaveScan([TiFilterDeg30(), TiFilterDeg120()]);

%NESTED PROCEDURE: PLOT AND SAVE SCAN
%PARAMETERS:
  %scanArray: array of a scan object at different angles
function plotAndSaveScan(scanArray)
  
  n = numel(scanArray); %number of angles
  scan = scanArray(1); %use the first scan to get the calibration images
  imageArray = zeros(scan.height, scan.width, n); %declare array for storing the x-ray images
  %get name of the dataset, get it from the superclass name
  dataName = superclasses(scan);
  dataName = dataName{1};
  
  %load each x-ray image
  for i = 1:n
    imageArray(:,:,i) = scanArray(i).loadImage(1);
  end

  %get the clim using the min and max of the x-ray images
  %the calibration clim will use the min over all calibration images
  clim = [min(min(min(imageArray))), max(max(max(imageArray)))];
  
  %for each angle
  for i = 1:n
    %plot the x-ray image
    fig = LatexFigure.sub();
    imagescPlot = Imagesc(imageArray(:,:,i));
    imagescPlot.setCLim(clim);
    imagescPlot.plot();
    saveas(fig,fullfile('reports','figures','data', ...
        strcat(class(scanArray(i)),'.eps')),'epsc');
    %plot the x-ray simulation
    fig = LatexFigure.sub();
    imagescPlot = Imagesc(scanArray(i).getArtistImage());
    imagescPlot.setCLim(clim);
    imagescPlot.plot();
    saveas(fig,fullfile('reports','figures','data', ...
        strcat(class(scanArray(i)),'_sim','.eps')),'epsc');
  end
  
  %for each power
  for i = 1:scan.whiteIndex()
    
    %get the calibration scan image
    calibration = scan.calibrationScanArray(i).loadImage(1);
    
    %plot the calibration image
    fig = LatexFigure.sub();
    imagescPlot = Imagesc(calibration);
    imagescPlot.plot();
    saveas(fig,fullfile('reports','figures','data', ...
        strcat(dataName,'_calibration',num2str(i),'.eps')),'epsc');
    
    %plot the calibration simulation
    %calibration simulation using power of 0 are not obtained so are not used
    if (i >= 2)
      calibration = scan.calibrationScanArray(i).getArtistImage();
      fig = LatexFigure.sub();
      imagescArtistPlot = Imagesc(calibration);
      imagescArtistPlot.setCLim(imagescPlot.clim);
      imagescArtistPlot.plot();
      saveas(fig,fullfile('reports','figures','data', ...
        strcat(dataName,'_calibrationSim',num2str(i),'.eps')),'epsc');
    end
    
  end

end