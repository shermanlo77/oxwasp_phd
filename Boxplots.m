%MIT License
%Copyright (c) 2019 Sherman Lo

%CLASS: BOXPLOTS
%Custom box plot class for plotting multiple box plots
classdef Boxplots < handle
  
  %MEMBER VARIABLES
  properties (SetAccess = private)
    boxplotArray; %cell array of boxplots
    nBoxplot; %number of boxplots
    wantTrend = false; %boolean, show trend line
    trendLine; %graph object
  end
  
  methods (Access = public)
    
    %CONSTRUCTOR
    %PARAMETERS:
      %X: matrix, dim 1: for each obversation, dim 2: for each group or boxplot
    function this = Boxplots(X)
      %get the number of boxplots
      [~,this.nBoxplot] = size(X);
      %create an array of box plots
      this.boxplotArray = cell(1,this.nBoxplot);
      %for each boxplot
      for i = 1:this.nBoxplot
        %instantise a boxplot and save it in this.boxplotArray
        this.boxplotArray{i} = this.getBoxplot(X(:,i));
        %set the position of the current boxplot
        this.boxplotArray{i}.setPosition(i - 1);
      end
    end
    
    %METHOD: PLOT
    %Plot each boxplot
    function plot(this)
      for i = 1:this.nBoxplot
        this.boxplotArray{i}.plot();
      end
      %if want the trend line, plot it
      if this.wantTrend
        x = zeros(this.nBoxplot, 1);
        y = zeros(this.nBoxplot, 1);
        for i = 1:this.nBoxplot
          x(i) = this.boxplotArray{i}.position;
          y(i) = this.boxplotArray{i}.median;
        end
        this.trendLine = line(x,y);
        this.trendLine.Color = this.boxplotArray{1}.colour;
      end
      %increment the colour order index
      ax = gca;
      ax.ColorOrderIndex = ax.ColorOrderIndex+1;
      if (ax.ColorOrderIndex > numel(ax.ColorOrder(:,1)))
        ax.ColorOderIndex = 1;
      end
    end
    
    %METHOD: SET POSITION
      %Set the position of each boxplot
    %PARAMETERS:
      %position: vector of positions
    function setPosition(this, position)
      for i = 1:this.nBoxplot
        this.boxplotArray{i}.setPosition(position(i));
      end
    end
    
    %METHOD: SET WANT OUTLIER
    function setWantOutlier(this, wantOutlier)
      for i = 1:this.nBoxplot
        this.boxplotArray{i}.setWantOutlier(wantOutlier);
      end
    end
    
    %METHOD: SET WANT TREND
    function setWantTrend(this, wantTrend)
      this.wantTrend = wantTrend;
    end
    
    %METHOD: SET WANT MEDIAN
    function setWantMedian(this, wantMedian)
      for i = 1:this.nBoxplot
        this.boxplotArray{i}.setWantMedian(wantMedian);
      end
    end
    
    %METHOD: SET COLOUR
      %Set the colour of each boxplot
    %PARAMETERS:
      %colour: colour of the boxplot
    function setColour(this, colour)
      for i = 1:this.nBoxplot
        this.boxplotArray{i}.setColour(colour);
      end
    end
    
    %METHOD: SET WHISKER CAP SIZE
      %Set the size of the whisker cap
    %PARAMETERS:
      %whiskerCapSize: size of the whisker cap
    function setWhiskerCapSize(this,whiskerCapSize)
      for i = 1:this.nBoxplot
        this.boxplotArray{i}.setWhiskerCapSize(whiskerCapSize);
      end
    end
    
    %METHOD: SET OUTLIER MARK
      %Set the mark for outliers
    %PARAMETERS:
      %outlierMark: mark for outliers
    function setOutlierMark(this,outlierMark)
      for i = 1:this.nBoxplot
        this.boxplotArray{i}.setOutlierMark(outlierMark);
      end
    end
    
    %METHOD: SET OUTLIER SIZE
    %Set the size for the outlier mark
    %PARAMETERS:
    %outlier_size: size of the outlier mark
    function setOutlierSize(this,outlierSize)
      for i = 1:this.nBoxplot
        this.boxplotArray{i}.setOutlierSize(outlierSize);
      end
    end
    
    %METHOD: GET LEGEND AXIS
      %Return axis object for the purpose of legend plotting
    function ax = getLegendAx(this)
      ax = this.boxplotArray{1}.legendAx;
    end
    
    %METHOD: SET TO BW
    function setToBw(this)
      for i = 1:this.nBoxplot
        this.boxplotArray{i}.setToBw();
      end
    end
    
  end
  
  methods (Access = protected)
    
    %METHOD: GET BOXPLOT
      %Return a boxplot object given data X
    function boxplot = getBoxplot(this, X)
      boxplot = Boxplot(X);
    end
    
  end
  
end

