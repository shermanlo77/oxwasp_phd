%MIT License
%Copyright (c) 2019 Sherman Lo

%CLASS: HISTOGRAM 3D HEATMAP
%Plots a 2D histogram using a heatmap
%
%How to use:
  %Instantise object, set the member variables to change the options
  %Call the method plot
classdef Hist3Heatmap < handle
  
  properties (SetAccess = public)
    isLog = true; %show heatmap as log value
    nBin = [100,100]; %number of bins in [y,x] direction
    %truncate the display of the heatmap as a percentage, [y,x] direction
    percentageCapture = [0.99,0.99];
    nInterpolate = 50; %for poster version
  end
  
  methods (Access = public)
    
    %CONSTRUCTOR
    function this = Hist3Heatmap()
    end
    
    %METHOD: PLOT
    %Plot the histogram heatmap
    %PARAMETERS:
      %x: column vector
      %y: column vector
    function ax = plot(this,x,y)
      %bin the data
      [N,c] = hist3([y,x],this.nBin);
      %normalize N so that the colormap is the frequency density
      if this.isLog
        ax = this.plotMap(cell2mat(c(2)),cell2mat(c(1)), ...
            log10(N/( (c{2}(2)-c{2}(1))*(c{1}(2)-c{1}(1)) ) ) );
      else
        ax = this.plotMap(cell2mat(c(2)),cell2mat(c(1)), ...
            N/( (c{2}(2)-c{2}(1))*(c{1}(2)-c{1}(1)) )  );
      end
    end
    
  end
  
  methods (Access = private)
    
    %METHOD: PLOT
    %Plot image
    function ax = plotMap(this,x,y,z)
      ax = axes;
      imagesc(x, y, z);
      axis xy; %switch the y axis
      %truncate the axis
      xP = (1-this.percentageCapture(2))/2;
      ax.XLim = quantile(x,[xP,1-xP]);
      yP = (1-this.percentageCapture(1))/2;
      ax.XLim = quantile(x,[yP,1-yP]);
      colorbar;
    end
    
    %METHOD: POSTER PLOT
    %Uses bigger squares suitable for larger poster images
    function ax = posterPlot(this,x,y)
      this.checkParameters(x,y);
      %bin the data
      [N,c] = hist3([y,x],this.nBin);
      x = cell2mat(c(2));
      x = linspace(x(1),x(end),this.nBin(2)*this.nInterpolate);
      y = cell2mat(c(1));
      y = linspace(y(1),y(end),this.nBin(1)*this.nInterpolate);
      if this.isLog
        z = log10(N/( (c{2}(2)-c{2}(1))*(c{1}(2)-c{1}(1))));
      else
        z = N/( (c{2}(2)-c{2}(1))*(c{1}(2)-c{1}(1)));
      end
      [x_grid,y_grid] = meshgrid(x,y);
      z = interp2(cell2mat(c(2)),cell2mat(c(1)),z,x_grid,y_grid,'nearest');
      ax = this.plotMap(x,y,z);
    end
    
  end
  
end

