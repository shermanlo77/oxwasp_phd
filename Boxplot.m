%CLASS BOXPLOT
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
        outlier_index; %boolean vector, true for outlier data
        
        position; %scalar, x position of the box plot
        colour; %colour of the box plot
        
        want_whisker_cap; %boolean, true if want whisker cap
        whisker_cap_size; %size of the whisker cap
        
        outlier_colour; %colour of the outlier mark
        outlier_mark; %mark of outlier
        outlier_size; %size of the outlier mark
    end
    
    %METHODS
    methods (Access = public)
        
        %CONSTURCTOR
        function this = Boxplot(X)
            %assign member variables
            this.X = X;
            %assign default values to member variables
            this.position = 0;
            this.colour = [0,0,1];
            this.want_whisker_cap = true;
            this.whisker_cap_size = 6;
            this.outlier_colour = [1,0,0];
            this.outlier_mark = 'x';
            this.outlier_size = 3;
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
        
        %METHOD: SET WHISKER CAP
        %Set the whisker cap to be on of off
        %PARAMETERS:
            %want_whisker_cap: true if want whisker cap on
        function setWhiskerCap(this,want_whisker_cap)
            this.want_whisker_cap = want_whisker_cap;
        end
        
        %METHOD: SET WHISKER CAP SIZE
        %Set the size of the whisker cap
        %PARAMETERS:
            %whisker_cap_size: size of the whisker cap
        function setWhiskerCapSize(this,whisker_cap_size)
            this.whisker_cap_size = whisker_cap_size;
        end
        
        %METHOD: SET OUTLIER COLOUR
        %Set the colour of the outlier
        %PARAMETERS:
            %colour: colour of the outlier
        function setOutlierColour(this,colour)
            this.outlier_colour = colour;
        end
        
        %METHOD: SET OUTLIER MARK
        %Set the mark for outliers
        %PARAMETERS:
            %outlier_mark: mark for outliers
        function setOutlierMark(this,outlier_mark)
            this.outlier_mark = outlier_mark;
        end
        
        %METHOD: SET OUTLIER SIZE
        %Set the size for the outlier mark
        %PARAMETERS:
            %outlier_size: size of the outlier mark
        function setOutlierSize(this,outlier_size)
            this.outlier_size = outlier_size;
        end
        
        %METHOD: PLOT
        %Plot the box plot
        function plot(this)
            %set all the required statistics and save it in the member variables
            this.getQuartiles();
            this.getOutlier();
            this.getWhisker();

            %plot outliers
            n_outlier = sum(this.outlier_index);
            outlier = line(ones(1,n_outlier)*this.position, this.X(this.outlier_index)');
            outlier.LineStyle = 'none';
            outlier.Marker = this.outlier_mark;
            outlier.Color = this.outlier_colour;
            outlier.MarkerSize = this.outlier_size;
            
            %draw the whisker
            whisker_line = line([this.position,this.position],this.whisker);
            whisker_line.Color = this.colour;
            
            %if want whisker cap
            if this.want_whisker_cap
                %draw an arrow at the end of the whiskers
                whisker_cap_upper = line(this.position,this.whisker(2));
                whisker_cap_upper.Marker = '^';
                whisker_cap_upper.Color = this.colour;
                whisker_cap_upper.MarkerFaceColor = this.colour;
                whisker_cap_upper.MarkerSize = this.whisker_cap_size;
                
                whisker_cap_lower = line(this.position,this.whisker(1));
                whisker_cap_lower.Marker = 'v';
                whisker_cap_lower.Color = this.colour;
                whisker_cap_lower.MarkerFaceColor = this.colour;
                whisker_cap_lower.MarkerSize = this.whisker_cap_size;
            end
            
            %draw the box
            box = line([this.position,this.position],this.quartiles);
            box.LineWidth = 4;
            box.Color = this.colour;
            
            %draw soild circle at median
            median_outer = line(this.position,this.median);
            median_outer.LineStyle = 'none';
            median_outer.Marker = 'o';
            median_outer.MarkerFaceColor = [1,1,1];
            median_outer.Color = this.colour;
            
            %draw point at median
            median_inner = line(this.position,this.median);
            median_inner.LineStyle = 'none';
            median_inner.Marker = '.';
            median_inner.Color = this.colour;
            
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
            this.whisker(1) = min(this.X(~this.outlier_index));
            this.whisker(2) = max(this.X(~this.outlier_index));
        end

        %METHOD: GET OUTLIER
        %Set which data are outliers or not, save the boolean in the member variable outlier_index
        function getOutlier(this)
            iqr = this.quartiles(2) - this.quartiles(1);
            this.outlier_index = (this.X < this.quartiles(1) - 1.5 * iqr) | (this.X > this.quartiles(2) + 1.5 * iqr);
        end
        
    end
    
end
