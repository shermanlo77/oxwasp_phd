classdef EmpericalConvolution < handle
    
    %MEMBER VARIABLES
    properties (SetAccess = private)
        n_col; %number of columns to sample
        n_row; %number of rows to sample
        kernel_size; %2 column vector [height, width] of the size of the moving window
        z_image; %image of z statistics
        
        mean_null; %image of emperical null mean parameter
        var_null; %image of emperical null var parameter
        p_image; %image of p values
        sig_image; %boolean image of significant pixels
    end
    
    %METHODS
    methods (Access = public)
        
        %CONSTRUCTOR
        %PARAMETERS:
            %z_image: image of z statistics
            %n_row: number of rows to sample
            %n_col: number of columns to sample
            %kernel_soze: 2 column vector [height, width] of the size of the moving window
        function this = EmpericalConvolution(z_image, n_row, n_col, kernel_size)
            %assign member variables
            this.z_image = z_image;
            this.n_row = n_row;
            this.n_col = n_col;
            this.kernel_size = kernel_size;
        end
        
        function estimateNull(this, n_linspace)
            
            %get the size of the z image
            [height, width] = size(this.z_image);
            
            %get array of x and y (or column and row index)
            %they are equally spaced out points within the image
            x_array = round(linspace(1,width,this.n_col));
            y_array = round(linspace(1,height,this.n_row));
            
            %declare variables which are of size n_row x n_col
            this.mean_null = zeros(this.n_row, this.n_col);
            this.var_null = zeros(this.n_row, this.n_col);
            
            %do a meshgrid
            [x_grid, y_grid] = meshgrid(x_array,y_array);
            
            kernel_half_size = round(this.kernel_size / 2);
            
            %for each column
            for i_col = 1:this.n_col
                
                %get the x coordinate
                x_center = x_array(i_col);
                
                %get the x_index of the window
                x_window = (x_center - kernel_half_size(2)) : (x_center + kernel_half_size(2));
                %check the bounary of the window
                x_window(x_window<1) = [];
                x_window(x_window>width) = [];
                
                %for each row
                for i_row = 1:this.n_row
                    
                    %get the y_coordinate
                    y_center = y_array(i_row);
                    
                    %get the y_index of the window
                    y_window = (y_center - kernel_half_size(1)) : (y_center + kernel_half_size(1));
                    %check the bounary of the window
                    y_window(y_window<1) = [];
                    y_window(y_window>height) = [];
                    
                    %instantise a z tester
                    z_tester = ZTester(this.z_image(y_window,x_window));
                    %get the emperical null
                    z_tester.estimateNull(n_linspace);
                    
                    %get the p value in the middle of the window
                    this.mean_null(i_row, i_col) = z_tester.mean_null;
                    this.var_null(i_row, i_col) = (z_tester.std_null)^2;
                    
                end
            end
            
            %meshgrid for 2d interpolation
            [x_full, y_full] = meshgrid(1:width, 1:height);
            %interpolate the image
            this.mean_null = interp2(x_grid,y_grid,this.mean_null,x_full,y_full);
            this.var_null = interp2(x_grid,y_grid,this.var_null,x_full,y_full);
        end
        
        function setMask(this, segmentation)
            this.mean_null(~segmentation) = nan;
            this.var_null(~segmentation) = nan;
        end
        
        function doTest(this)
            
            z_null = (this.z_image - this.mean_null) ./ sqrt(this.var_null);
            z_tester = ZTester(z_null);
            z_tester.doTest();
            this.p_image = z_tester.p_image;
            this.sig_image = z_tester.sig_image;
            
        end
        
    end
    
end

