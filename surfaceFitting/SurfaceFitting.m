classdef SurfaceFitting < handle
    %SURFACEFITTING Summary of this class goes here
    %   Detailed explanation goes here
    
    %MEMBER VARIABLES
    properties
        
        total_size; %two vector [height, width] representing the total size of the detector image in pixels
        active_size; %two vector [height, width] representing the total size of the cropped image in pixels
        active_area; %active_size(1)*active_size(2), area of the obtained (cropped image)
        n_panel_column; %scalar representing the number of columns of 2 panels in the detector
        
        panel_height; %height of each panel in pixels
        panel_width; %width of each non-border panel in pixels
        panel_width_edge; %width of panels on the edges (far left and far right) in pixels
        
        loaded_white; %boolean indicating bw_stack is white image if true, otherwise black
        bw_stack; %array of black or white images
        n_bw; %number of images in bw_stack
        
        black_train; %a black image
        black_fit; %smoothed black image
        white_train; %a white image
        white_fit; %smoothed white image
        
    end
    
    methods
        
        %CONSTRUCTOR
        %PARAMETERS
            %total_size: [height, width] of the size of the detector image
            %active_size: [height, width] of the size of the cropped image
            %n_panel: [height, width] of the number of panels in the image
        function this = SurfaceFitting(total_size,active_size,n_panel_column)
            %assign member variables
            this.total_size = total_size;
            this.active_size = active_size;
            this.active_area = active_size(1)*active_size(2);
            this.n_panel_column = n_panel_column;
            
            %panel height (there are 2 rows of panels)
            this.panel_height = this.active_size(1)/2;
            %panel width (divide the width by the number of columns of panels)
            this.panel_width = this.total_size(2)/n_panel_column;
            %panel width at the edges (from cropping)
            this.panel_width_edge = this.panel_width - ((this.total_size(2)-this.active_size(2))/2);
            
            %declare empty matrices
            this.black_fit = zeros(active_size);
            this.white_fit = zeros(active_size);
        end
        
        %LOAD BLACK IMAGES
        function loadBlack(this,file_location)
            this.loaded_white = false; %set boolean to false
            %save the black images to this.bw_stack
            [this.bw_stack,~,~,this.n_bw] = load_black(file_location);
        end
        
        %LOAD WHITE IMAGES
        function loadWhite(this,file_location)
            this.loaded_white = true; %set boolean to true
            %save the black images to this.bw_stack
            [this.bw_stack,~,~,this.n_bw] = load_white(file_location);
        end
        
        %FIT POLYNOMIAL SURFACE
        %Fit polynomial surface with order p
        %PARAMETERS:
            %x: vector of x coordinates
            %y: vector of y coordinates
            %z: vector of greyvalues
            %p: order of polynomial
        %RETURN:
            %sfit_obj: sfit object (surface fit)
        function sfit_obj = fitPolynomialSurface(this,x,y,z,p)
            polynomial_string = strcat('poly',num2str(p),num2str(p));
            sfit_obj = fit([x,y],z,polynomial_string);
        end
        
        %FIT POLYNOMIAL PANEL
        %Using an image from the stack of black/white images, fit polynomial
        %surface to it.
        %PARAMETERS:
            %train_index: the image from this.black_stack to be used
            %p: order of the polynomial
        function fitPolynomialPanel(this,train_index,p)
            
            %get the black/white image from the stack
            bw_train = this.bw_stack(:,:,train_index);
            
            %save bw_train as the training image for either black or white
            if this.loaded_white
                this.white_train = bw_train;
            else
                this.black_train = bw_train;
            end
            
            %for each column
            for i_column = 1:this.n_panel_column
                
                %for each row
                for i_row = 1:2
                    
                    %get the range of rows which covers a panel
                    height_range = (1 + (i_row-1)*this.panel_height) : (i_row*this.panel_height);
                    
                    %then get the range of columns which covers that panel
                    %special case if the column is on the boundary
%                     if i_column == 1
%                         width_range = 1:(this.panel_width_edge);
%                     elseif i_column == this.n_panel_column
%                         width_range = (this.active_size(2)-this.panel_width_edge+1):this.active_size(2);
%                     %ordinary case:
%                     else
%                         width_range = (this.panel_width_edge + (i_column-2)*this.panel_width + 1) : (this.panel_width_edge + (i_column-1)*this.panel_width);
%                     end

                    if i_column ~= this.n_panel_column
                        width_range = ((i_column-1)*this.panel_width+1):(i_column*this.panel_width);
                    else
                        width_range = ((i_column-1)*this.panel_width+1):(this.active_size(2));
                    end
                    
                    %for the given panel, get the grey values
                    z_grid = bw_train(height_range,width_range);
                    
                    %obtain the range of x and y in grid form
                    [x_grid,y_grid] = meshgrid(1:numel(width_range),1:numel(height_range));
                    %convert x,y,z from grid form to vector form
                    x = reshape(x_grid,[],1);
                    y = reshape(y_grid,[],1);
                    z = reshape(z_grid,[],1);
                    
                    %scale z to have zero mean and std 1
                    shift = mean(z);
                    scale = std(z);
                    z = (z-shift)/scale;
                    
                    %fit polynomial surface to the data (x,y,z)
                    sfit_obj = this.fitPolynomialSurface(x,y,z,p);
                    
                    %save the surface of the panel to this.black_fit or this.white_fit
                    if this.loaded_white
                        this.white_fit(height_range,width_range) = (sfit_obj(x_grid,y_grid)*scale)+shift;
                    else
                        this.black_fit(height_range,width_range) = (sfit_obj(x_grid,y_grid)*scale)+shift;
                    end
                    
                end
                
            end
            
        end
        
        %CLEAR BLACK WHITE STACK
        %Assign empty matrix to this.bw_stack
        function clearBWStack(this)
            this.bw_stack = [];
        end
        
        %PLOT POLYNOMIAL PANEL (BLACK)
        %Plot heatmap of the black image, smoothed and unsmoothed
        %PARAMETERS:
            %percentile: two vector, defines the limits of the greyvalues in the heatmap
        function plotPolynomialPanel_black(this,percentile)
            %get the percentiles of the greyvalues
            clims = prctile(reshape(this.black_train,[],1),percentile);
            %heatmap plot the unsmooth data
            
            figure;
            imagesc(this.black_train,clims);
            colorbar;
            colormap gray;
            %heatmap plot the smooth data
            
            figure;
            imagesc(this.black_fit,clims);
            colorbar;
            colormap gray;
        end
        
        %PLOT POLYNOMIAL PANEL (WHITE)
        %Plot heatmap of the white image, smoothed and unsmoothed
        %PARAMETERS:
            %percentile: two vector, defines the limits of the greyvalues in the heatmap
        function plotPolynomialPanel_white(this,percentile)
            %get the percentiles of the greyvalues
            clims = prctile(reshape(this.white_train,[],1),percentile);
            %heatmap plot the unsmooth data
            figure;
            imagesc(this.white_train,clims);
            colorbar;
            colormap gray;
            %heatmap plot the smooth data
            figure;
            imagesc(this.white_fit,clims);
            colorbar;
            colormap gray;
        end
        
        %MEAN SQUARED ERROR
        %Calculate the mean squared difference between the smoothed
        %image and a training image from this.bwk_stack
        %PARAMETER:
            %index: index pointing to an image in this.bw_stack
        %RETURN:
            %mse: mean squared error
        function mse = meanSquaredError(this,index)
            %calculate the mean squared error
            mse = sum(sum((this.getResidualImage(index)).^2))/this.active_area;
        end
        
        %GET RESIUDAL IMAGE
        %Get image of fitted polynomial image subtract a bw image, selected
        %by the parameter index
        function residual_image = getResidualImage(this,index)
            %get the test image
            test_image = this.bw_stack(:,:,index);
            %load the smoothed image 
            if this.loaded_white
                fit_image = this.white_fit;
            else
                fit_image = this.black_fit;
            end
            %work out the difference between the fitted and test image
            residual_image = test_image-fit_image;
        end
        
        %CROSS VALIDATION
        %Fit polynomials on the train_index-th image from this.bw_stack
        %with order 1,2,3,4,5. For each order, calculate the mean squared
        %error
        function mse_array = crossValidation(this,train_index)
            %declare array of mse for each polynomial order
            mse_array = zeros(1,5);
            %for each polynomial order
            for p = 1:5
                %fit the polynomial
                this.fitPolynomialPanel(train_index,p);
                %declare array of mse for each test image
                mse_sample = zeros(1,this.n_bw-1);
                %count the number of test images gone through so far
                sample_index = 1;
                %for each image in this.bw_stack
                for i_sample = 1:this.n_bw
                    %if this image is not the training image
                    if i_sample ~= train_index
                        %calculate the mean squared error and save it to mse_sample
                        mse_sample(sample_index) = this.meanSquaredError(i_sample);
                        %increment sample_index
                        sample_index = sample_index + 1;
                    end
                end
                %save the sample mean of the mse over all test images
                mse_array(p) = mean(mse_sample);
            end
            
        end
        
        %ROTATE CROSS VALIDATION
        %Do cross validation using each image as a training set once.
        %RETURN:
            %mse_array: this.bw x 5 matrix containing the mse
        function mse_array = rotateCrossValidation(this)
            %declare array of mse for each polynomial order
            mse_array = zeros(this.n_bw,5);
            %for each image
            for i_data_index = 1:this.n_bw
                disp(i_data_index);
                %get the mse for each order
                mse_array(i_data_index,:) = this.crossValidation(i_data_index);
            end         
        end
        
    end
    
end

