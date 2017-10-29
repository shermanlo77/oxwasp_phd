classdef BoxplotArray < handle

    properties
        
        boxplot_array
        n_boxplot;
        
    end
    
    methods
        
        function this = BoxplotArray(X)
            [~,this.n_boxplot] = size(X);
            this.boxplot_array = cell(1,this.n_boxplot);
            for i_group = 1:this.n_boxplot
                this.boxplot_array{i_group} = Boxplot(X(:,i_group));
                this.boxplot_array{i_group}.position = i_group - 1;
            end
        end
        
        function plotBoxplot(this)
            for i_group = 1:this.n_boxplot
                this.boxplot_array{i_group}.plotBoxplot();
            end
        end
        
        
    end
    
end

