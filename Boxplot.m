classdef Boxplot < handle
    
    properties
        X;
        median;
        quartiles;
        whisker;
        outlier_index;
        
        position;
        colour;
    end
    
    methods
        
        function this = Boxplot(X)
            this.X = X;
            this.position = 0;
            this.colour = [0,0,1];
        end
        
        function getQuantiles(this)
           q = quantile(this.X,[0.25,0.5,0.75]);
           this.quartiles = zeros(1,2);
           this.quartiles(1) = q(1);
           this.median = q(2);
           this.quartiles(2) = q(3);
        end
        
        function getWhisker(this)
            iqr = this.quartiles(2) - this.quartiles(1);
            this.outlier_index = (this.X < this.quartiles(1) - 1.5 * iqr) | (this.X > this.quartiles(2) + 1.5 * iqr);
            this.whisker = zeros(1,2);
            this.whisker(1) = min(this.X(~this.outlier_index));
            this.whisker(2) = max(this.X(~this.outlier_index));
        end
        
        function plotBoxplot(this)
            this.getQuantiles();
            this.getWhisker();

            whisker_line = line([this.position,this.position],this.whisker);
            whisker_line.Color = this.colour;
            
            box = line([this.position,this.position],this.quartiles);
            box.LineWidth = 4;
            box.Color = this.colour;
            
            n_outlier = sum(this.outlier_index);
            outlier = line(ones(1,n_outlier)*this.position, this.X(this.outlier_index)');
            outlier.LineStyle = 'none';
            outlier.Marker = 'x';
            outlier.Color = this.colour;
            
            median_outer = line(this.position,this.median);
            median_outer.LineStyle = 'none';
            median_outer.Marker = 'o';
            median_outer.MarkerFaceColor = [1,1,1];
            median_outer.Color = this.colour;
            
            median_inner = line(this.position,this.median);
            median_inner.LineStyle = 'none';
            median_inner.Marker = '.';
            median_inner.Color = this.colour;
            
        end
        
    end
    
end

