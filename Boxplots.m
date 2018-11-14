%CLASS: BOXPLOTS
%Custom box plot class for plotting multiple box plots
classdef Boxplots < handle

    %MEMBER VARIABLES
    properties (SetAccess = private)
        boxplot_array; %cell array of boxplots
        n_boxplot; %number of boxplots
    end
    
    methods (Access = public)
    
        %CONSTRUCTOR
        %PARAMETERS:
            %X: matrix, dim 1: for each obversation, dim 2: for each group or boxplot
            %is_standard: boolean, true to use standard box plot, false to use generalised box plot
        function this = Boxplots(X, is_standard)
            %get the number of boxplots
            [~,this.n_boxplot] = size(X);
            %create an array of box plots
            this.boxplot_array = cell(1,this.n_boxplot);
            %for each boxplot
            for i_group = 1:this.n_boxplot
                %instantise a boxplot and save it in this.boxplot_array
                %according to is_standard, instantise either a standard or a generalised boxplot
                if is_standard
                    this.boxplot_array{i_group} = Boxplot(X(:,i_group));
                else
                    this.boxplot_array{i_group} = BoxplotGeneralised(X(:,i_group));
                end
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
        
        %METHOD: SET OUTLIER COLOUR
        %Set the colour of the outlier
        %PARAMETERS:
            %colour: colour of the outlier
        function setOutlierColour(this,colour)
            for i_group = 1:this.n_boxplot
                this.boxplot_array{i_group}.setOutlierColour(colour);
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
    
end

