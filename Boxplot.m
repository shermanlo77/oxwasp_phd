%MIT License
%Copyright (c) 2019 Sherman Lo

%CLASS: BOXPLOT
%Custom boxplot class, for plotting box plot given data
%
%Draws a thick line between the quartiles
%Draws whiskers between the min and max of non-outlier data
%Outlier data is defined using the quartiles and 1.5 IQR (standard)
%Draws a point at the median
classdef Boxplot < handle
  
  %PROPERTIES
  properties (SetAccess = protected)
    X; %data vector, dim 1: for each obversation
    median; %scalar, median
    quartiles; %2 vector, contains the 1st and 3rd quartile
    whisker; %2 vector, contains the min and max of non-outlier data
    outlierIndex; %boolean vector, true for outlier data
    
    position = 0; %scalar, x position of the box plot
    colour; %colour of the box plot
    
    wantOutlier = true;
    wantMedian = true;
    hasUpperOutlier; %boolean, true if has outliers bigger than the upper whisker
    hasLowerOutlier; %boolean, ture if has outliers smaller than the lower whisker
    whiskerCapSize = 6; %size of the whisker cap
    
    outlierMark = 'x'; %mark of outlier
    outlierSize = 4; %size of the outlier mark
    
    legendAx; %what to show in the legend
  end
  
  %METHODS
  methods (Access = public)
    
    %CONSTURCTOR
    %PARAMETERS:
      %X: column vector of data
    function this = Boxplot(X)
      %assign member variables
      this.X = X;
      ax = gca;
      this.colour = ax.ColorOrder(ax.ColorOrderIndex,:);
    end
    
    %METHOD: SET POSITION
      %Set the x position of the box plot
    %PARAMETERS:
      %position: x position of the box plot
    function setPosition(this,position)
      this.position = position;
    end
    
    %METHOD: SET COLOUR
      %Set the colour of the box plot
    %PARAMETERS:
      %colour: colour of the box plot
    function setColour(this,colour)
      this.colour = colour;
    end
    
    %METHOD: SET WANT OUTLIER
      %Set if want to show outliers or not
    function setWantOutlier(this, wantOutlier)
      this.wantOutlier = wantOutlier;
    end
    
    %METHOD: SET WANT MEDIAN
      %Set if want to show the median
    function setWantMedian(this, wantMedian)
      this.wantMedian = wantMedian;
    end
    
    %METHOD: SET WHISKER CAP SIZE
      %Set the size of the whisker cap
    %PARAMETERS:
      %whiskerCapSize: size of the whisker cap
    function setWhiskerCapSize(this,whiskerCapSize)
      this.whiskerCapSize = whiskerCapSize;
    end
    
    %METHOD: SET OUTLIER MARK
      %Set the mark for outliers
    %PARAMETERS:
      %outlierMark: mark for outliers
    function setOutlierMark(this,outlierMark)
      this.outlierMark = outlierMark;
    end
    
    %METHOD: SET OUTLIER SIZE
      %Set the size for the outlier mark
    %PARAMETERS:
      %outlierSize: size of the outlier mark
    function setOutlierSize(this,outlierSize)
      this.outlierSize = outlierSize;
    end
    
    %METHOD: PLOT
      %Plot the box plot
    function plot(this)
      %set all the required statistics and save it in the member variables
      this.getQuartiles();
      this.getOutlier();
      this.getWhisker();
      
      %plot outliers
      if (this.wantOutlier)
        nOutlier = sum(this.outlierIndex);
        outlier = line(ones(1,nOutlier)*this.position, this.X(this.outlierIndex)');
        if nOutlier ~= 0
          outlier.LineStyle = 'none';
          outlier.Marker = this.outlierMark;
          outlier.Color = this.colour;
          outlier.MarkerSize = this.outlierSize;
        end
      else
        if (this.hasUpperOutlier)
          %draw an arrow at the end of the whiskers
          whiskerCapUpper = line(this.position,this.whisker(2));
          whiskerCapUpper.Marker = '^';
          whiskerCapUpper.Color = this.colour;
          whiskerCapUpper.MarkerFaceColor = this.colour;
          whiskerCapUpper.MarkerSize = this.whiskerCapSize;
        elseif (this.hasLowerOutlier)
          whiskerCapLower = line(this.position,this.whisker(1));
          whiskerCapLower.Marker = 'v';
          whiskerCapLower.Color = this.colour;
          whiskerCapLower.MarkerFaceColor = this.colour;
          whiskerCapLower.MarkerSize = this.whiskerCapSize;
        end
      end
      
      %draw the whisker
      whiskerLine = line([this.position,this.position],this.whisker);
      whiskerLine.Color = this.colour;
      %the whisker is what to draw for the legend
      this.legendAx = whiskerLine;
      
      %draw the box
      box = line([this.position,this.position],this.quartiles);
      box.LineWidth = 4;
      box.Color = this.colour;
      
      %draw the median if requested
      if (this.wantMedian)
        
        %draw soild circle at median
        medianOuter = line(this.position,this.median);
        medianOuter.LineStyle = 'none';
        medianOuter.Marker = 'o';
        medianOuter.MarkerFaceColor = [1,1,1];
        medianOuter.Color = this.colour;
        
        %draw point at median
        medianInner = line(this.position,this.median);
        medianInner.LineStyle = 'none';
        medianInner.Marker = '.';
        medianInner.Color = this.colour;
        
      end
      
    end
    
  end
  
  methods (Access = protected)
    
    %METHOD: GET QUARTILES
    %Get the quartiles of the data and save it in the member variables
    function getQuartiles(this)
      q = quantile(this.X,[0.25,0.5,0.75]);
      this.quartiles = zeros(1,2);
      this.quartiles(1) = q(1);
      this.median = q(2);
      this.quartiles(2) = q(3);
    end
    
    %METHOD: GET WHISKER
    %Get the whiskers, that is the min and max of non-outlier data
    function getWhisker(this)
      this.whisker = zeros(1,2);
      this.whisker(1) = min(this.X(~this.outlierIndex));
      this.whisker(2) = max(this.X(~this.outlierIndex));
    end
    
    %METHOD: GET OUTLIER
    %Set which data are outliers or not, save the boolean in the member variable outlierIndex
    function getOutlier(this)
      iqr = this.quartiles(2) - this.quartiles(1);
      isUpperOutlier = (this.X > this.quartiles(2) + 1.5 * iqr);
      isLowerOutlier = (this.X < this.quartiles(1) - 1.5 * iqr);
      this.outlierIndex = isLowerOutlier | isUpperOutlier;
      this.hasUpperOutlier = any(isUpperOutlier);
      this.hasLowerOutlier = any(isLowerOutlier);
    end
    
  end
  
end
