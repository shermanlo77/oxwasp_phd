%ODD EVEN PLOT
%
%Plot the y profile for a given x for a black image
%Also plots for even y and odd y

clc;
close all;
clearvars;

scan = AbsNoFilterDeg30();
x = 879; %x position for the y profile
ylimAll = []; %y limit for graphs

seed = uint32(2146127166); %seed used for selecting random pixels for the interpolation graph
rng = RandStream('mt19937ar', 'Seed', seed); %random number generator
%rng used for selecting random replication for training shading correction, another for the
    %resulting shading correction
calibrationIndex = zeros(scan.calibrationScanArray(1).nSample, scan.whiteIndex);
for i = 1:scan.whiteIndex
  calibrationIndex(:,i) = rng.randperm(scan.calibrationScanArray(1).nSample)';
end

%for each shading correction
for i = 1:3
  
  %apply shading correction
  if (i==2)
    scan.addShadingCorrectorBw([calibrationIndex(1,1), calibrationIndex(1,end)]);
  elseif (i==3)
    scan.addShadingCorrectorLinear(1:scan.whiteIndex, calibrationIndex(1,:));
  end

  %get a column from the black image
  yProfile = scan.calibrationScanArray(1).loadImage(calibrationIndex(2,1));
  yProfile = yProfile(:,x);
  yCoordinate = 1:scan.height;
  
  %plot the y profile
  fig = LatexFigure.sub();
  plot(yCoordinate, yProfile);
  ylabel('grey value');
  xlabel('y coordinate');
  %use the ylim for all graphs
  if (i==1)
    ax = gca;
    ylimAll = ax.YLim;
  end
  ylim(ylimAll);
  saveas(fig,fullfile('reports','figures','data', ...
        strcat(mfilename,'1_',scan.getShadingCorrectionStatus(),'.eps')),'epsc');
  
  %plot the y profile for even y and odd y
  fig = LatexFigure.sub();
  plot(yCoordinate(1:2:end), yProfile(1:2:end));
  hold on;
  plot(yCoordinate(1:2:end), yProfile(2:2:end));
  legend('odd y', 'even y');
  ylabel('grey value');
  xlabel('y coordinate');
  ylim(ylimAll);
  saveas(fig,fullfile('reports','figures','data', ...
        strcat(mfilename,'2_',scan.getShadingCorrectionStatus(),'.eps')),'epsc');
  
end