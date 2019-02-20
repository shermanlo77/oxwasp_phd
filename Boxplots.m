%CLASS: BOXPLOTS
%Custom box plot class for plotting multiple box plots
classdef Boxplots < handle

    %MEMBER VARIABLES
    properties (SetAccess = private)
        boxplot_array; %cell array of boxplots
        n_boxplot; %number of boxplots
        wantTrend = false; %boolean, show trend line
        trendLine;
    end
    
    methods (Access = public)
    
        %CONSTRUCTOR
        %PARAMETERS:
            %X: matrix, dim 1: for each obversation, dim 2: for each group or boxplot
        function this = Boxplots(X, ~)
            %get the number of boxplots
            [~,this.n_boxplot] = size(X);
            %create an array of box plots
            this.boxplot_array = cell(1,this.n_boxplot);
            %for each boxplot
            for i_group = 1:this.n_boxplot
                %instantise a boxplot and save it in this.boxplot_array
                this.boxplot_array{i_group} = this.getBoxplot(X(:,i_group));
                %set the position of the current boxplot
                this.boxplot_array{i_group}.setPosition(i_group - 1);
            end
        end
        
        %METHOD: PLOT
        %Plot each boxplot
        function plot(this)
            for i_group = 1:this.n_boxplot
                this.boxplot_array{i_group}.plot();
            end
            %if want the trend line, plot it
            if this.wantTrend
              x = zeros(this.n_boxplot, 1);
              y = zeros(this.n_boxplot, 1);
              for i = 1:this.n_boxplot
                x(i) = this.boxplot_array{i}.position;
                y(i) = this.boxplot_array{i}.median;
              end
              this.trendLine = line(x,y);
              this.trendLine.Color = this.boxplot_array{1}.colour;
            end
        end
        
        %METHOD: SET POSITION
        %Set the position of each boxplot
        %PARAMETERS:
            %position: vector of positions
        function setPosition(this, position)
            for i_group = 1:this.n_boxplot
                this.boxplot_array{i_group}.setPosition(position(i_group));
            end
        end
        
        %METHOD: SET WANT OUTLIER
        function setWantOutlier(this, wantOutlier)
          for i_group = 1:this.n_boxplot
            this.boxplot_array{i_group}.setWantOutlier(wantOutlier);
          end
        end
        
        %METHOD: SET WANT TREND
        function setWantTrend(this, wantTrend)
          this.wantTrend = wantTrend;
        end
        
        %METHOD: SET WANT MEDIAN
        function setWantMedian(this, wantMedian)
          for i_group = 1:this.n_boxplot
            this.boxplot_array{i_group}.setWantMedian(wantMedian);
          end
        end
        
        %METHOD: SET COLOUR
        %Set the colour of each boxplot
        %PARAMETERS:
            %colour: colour of the boxplot
        function setColour(this, colour)
            for i_group = 1:this.n_boxplot
                this.boxplot_array{i_group}.setColour(colour);
            end
        end
        
        %METHOD: SET WHISKER CAP
        %Set the whisker cap to be on of off
        %PARAMETERS:
            %want_whisker_cap: true if want whisker cap on
        function setWhiskerCap(this,want_whisker_cap)
            for i_group = 1:this.n_boxplot
                this.boxplot_array{i_group}.setWhiskerCap(want_whisker_cap);
            end
        end
        
        %METHOD: SET WHISKER CAP SIZE
        %Set the size of the whisker cap
        %PARAMETERS:
            %whisker_cap_size: size of the whisker cap
        function setWhiskerCapSize(this,whisker_cap_size)
            for i_group = 1:this.n_boxplot
                this.boxplot_array{i_group}.setWhiskerCapSize(whisker_cap_size);
            end
        end
        
        %METHOD: SET OUTLIER MARK
        %Set the mark for outliers
        %PARAMETERS:
            %outlier_mark: mark for outliers
        function setOutlierMark(this,outlier_mark)
            for i_group = 1:this.n_boxplot
                this.boxplot_array{i_group}.setOutlierMark(outlier_mark);
            end
        end
        
        %METHOD: SET OUTLIER SIZE
        %Set the size for the outlier mark
        %PARAMETERS:
            %outlier_size: size of the outlier mark
        function setOutlierSize(this,outlier_size)
            for i_group = 1:this.n_boxplot
                this.boxplot_array{i_group}.setOutlierSize(outlier_size);
            end
        end
        
        %METHOD: GET LEGEND AXIS
        %Return axis object for the purpose of legend plotting
        function ax = getLegendAx(this)
          ax = this.boxplot_array{1}.legendAx;
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

