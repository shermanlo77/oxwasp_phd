classdef EmpericalConvolution_Parfor < EmpericalConvolution
    
    
    methods (Access = public)
        
        %CONSTRUCTOR
        %PARAMETERS:
            %z_image: image of z statistics
            %n_row: number of rows to sample
            %n_col: number of columns to sample
            %kernel_size: 2 column vector [height, width] of the size of the moving window
        function this = EmpericalConvolution_Parfor(z_image, n_row, n_col, kernel_size)
            this@EmpericalConvolution(z_image, n_row, n_col, kernel_size);
        end
        
        %METHOD: ESTIMATE NULL
            %Does a convolution of a sample of pixels
            %Estimates the parameters of the emperical null for each of these pixels
            %The blanks are then filled in using linear interpolation
            %The parameters of the null hypothesis are stored in the member variables mean_null and var_null
        function estimateNull(this, ~)
            
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
            
            this.z_tester_array = EmpericalConvolution_Parfor.estimateNullParfor(x_grid, y_grid, this.z_image, this.kernel_size);
            for i = 1:numel(x_grid)
                %get the emperical null parameters
                this.mean_null(i) = this.z_tester_array{i}.mean_null;
                this.var_null(i) = this.z_tester_array{i}.var_null;
            end
                    
            
            %meshgrid for 2d interpolation
            [x_full, y_full] = meshgrid(1:width, 1:height);
            %interpolate the image
            this.mean_null = interp2(x_grid,y_grid,this.mean_null,x_full,y_full);
            this.var_null = interp2(x_grid,y_grid,this.var_null,x_full,y_full);
            
            %get the variance null if we assume it is uniform
            this.setVarUniformNull();
        end
        
        
    end
    
    methods (Static)
        
        function z_tester_array = estimateNullParfor(x_grid, y_grid, z_image, kernel_size)
            [n_row, n_col] = size(x_grid);
            [height, width] = size(z_image);
            z_tester_array = cell(n_row,n_col);
            kernel_half_size = floor(kernel_size / 2);
            
            
            for i = 1:numel(x_grid)
                %get the x coordinate
                x_center = x_grid(i);
                
                %get the x_index of the window
                x_window = (x_center - kernel_half_size(2)) : (x_center + kernel_half_size(2));
                %check the bounary of the window
                x_window(x_window<1) = [];
                x_window(x_window>width) = [];
                    
                %get the y_coordinate
                y_center = y_grid(i);

                %get the y_index of the window
                y_window = (y_center - kernel_half_size(1)) : (y_center + kernel_half_size(1));
                %check the bounary of the window
                y_window(y_window<1) = [];
                y_window(y_window>height) = [];

                %instantise a z tester
                z_tester_array{i} = ZTester(z_image(y_window,x_window));
            end
            
            %for each column
            parfor i = 1:numel(x_grid)
                z_tester_array{i}.estimateNull();
            end
            
            
        end
        
        
    end
    
end