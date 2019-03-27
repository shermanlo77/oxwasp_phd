%CLASS: HISTOGRAM 3D HEATMAP

classdef Hist3Heatmap < handle
    
    properties (SetAccess = public)
        is_log;
        n_bin;
        percentage_capture;
        n_interpolate;
    end
    
    methods (Access = public)
        
        function this = Hist3Heatmap()
            this.is_log = true;
            this.n_bin = [100,100];
            this.percentage_capture = [0.99,0.99];
            this.n_interpolate = 50;
        end
        
        function ax = plot(this,x,y)
            this.checkParameters(x,y);
            %bin the data
            [N,c] = hist3([y,x],this.n_bin);
            %normalize N so that the colormap is the frequency density
            if this.is_log
                ax = this.plotMap(cell2mat(c(2)),cell2mat(c(1)), log10(N/( (c{2}(2)-c{2}(1))*(c{1}(2)-c{1}(1)) ) ) );
            else
                ax = this.plotMap(cell2mat(c(2)),cell2mat(c(1)), N/( (c{2}(2)-c{2}(1))*(c{1}(2)-c{1}(1)) )  );
            end 
        end
        
        function ax = plotMap(this,x,y,z)
            ax = axes;
            imagesc(x, y, z);
            axis xy; %switch the y axis
            x_p = (1-this.percentage_capture(1))/2;
            ax.XLim = quantile(x,[x_p,1-x_p]);
            y_p = (1-this.percentage_capture(2))/2;
            ax.XLim = quantile(x,[y_p,1-y_p]);
            colorbar;
        end
        
        function ax = posterPlot(this,x,y)
            this.checkParameters(x,y);
            %bin the data
            [N,c] = hist3([y,x],this.n_bin);
            x = cell2mat(c(2));
            x = linspace(x(1),x(end),this.n_bin(2)*this.n_interpolate);
            y = cell2mat(c(1));
            y = linspace(y(1),y(end),this.n_bin(1)*this.n_interpolate);
            
            
            if this.is_log
                z = log10(N/( (c{2}(2)-c{2}(1))*(c{1}(2)-c{1}(1))));
            else
                z = N/( (c{2}(2)-c{2}(1))*(c{1}(2)-c{1}(1)));
            end
            [x_grid,y_grid] = meshgrid(x,y);
            z = interp2(cell2mat(c(2)),cell2mat(c(1)),z,x_grid,y_grid,'nearest');
            ax = this.plotMap(x,y,z);
        end
        
    end
    
    methods (Access = private)
        
        function checkParameters(this, x, y)
            %check if sample_mean is a column vector, if not throw
            if ~iscolumn(x)
                error('Error in plotHistogramHeatmap(sample_mean,sample_var), sample_mean is not a column vector');
            end
            %check if sample_var is a column vector, if not throw
            if ~iscolumn(y)
                error('Error in plotHistogramHeatmap(sample_mean,sample_var), sample_var is not a column vector');
            end
            %check if sample_mean and sample_var has the same length, else throw
            n1 = numel(x);
            n2 = numel(y);
            if n1~=n2
                error('Error in plotHistogramHeatmap(sample_mean,sample_var), sample_mean and sample_var are not the same length');
            end
        end
    end
    
end

