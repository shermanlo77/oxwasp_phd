classdef SurfaceFitting < handle
    %SURFACEFITTING Fit and stores fit independent surfaces for each panel
    %   Stack of white, black and scan images are stored here. Surfaces are
    %   fitted to each panel independently. Subclasses implement its own
    %   method for fitting surfaces.
    
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
        
        scan_stack; %stack of images of the scan
        n_scan; %number of images in scan_stack
        
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
        
        %LOAD SCAN IMAGES
        function loadScan(this,file_location)
            [this.scan_stack,~,~,this.n_scan] = load_stack(file_location);
        end
        
        %FIT SURFACE
        %Using an image from the stack of black/white images, fit
        %surface to it.
        %PARAMETERS:
            %train_index: the image from this.black_stack to be used
            %p: parameter
        function fitSurface(this,train_index,p)
            
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
                    
                    %STANDARD CROP
%                     %get the range of rows which covers a panel
%                     height_range = (1 + (i_row-1)*this.panel_height) : (i_row*this.panel_height);

                    
                    %GUESS CROP
                    if i_row == 1
                        height_range = (1 + (i_row-1)*this.panel_height) : (i_row*this.panel_height+6);
                    else
                        height_range = (1 + (i_row-1)*this.panel_height+6) : (i_row*this.panel_height);
                    end

                    %STANDARD CROP
%                     %then get the range of columns which covers that panel
%                     %special case if the column is on the boundary
%                     if i_column == 1
%                         width_range = 1:(this.panel_width_edge);
%                     elseif i_column == this.n_panel_column
%                         width_range = (this.active_size(2)-this.panel_width_edge+1):this.active_size(2);
%                     %ordinary case:
%                     else
%                         width_range = (this.panel_width_edge + (i_column-2)*this.panel_width + 1) : (this.panel_width_edge + (i_column-1)*this.panel_width);
%                     end

                    %GUESS CROP
                    if i_column == 1
                        width_range = 1:(this.panel_width-2);
                    elseif i_column ~= this.n_panel_column
                        width_range = ((i_column-1)*this.panel_width+1-2):(i_column*this.panel_width-2);
                    else
                        width_range = ((i_column-1)*this.panel_width+1-2):(this.active_size(2));
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
                    
                    %fit surface to the data (x,y,z)
                    sfit_obj = this.fitPanel(x,y,z,p);
                    
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
        
        %PLOT BLACK SURFACE
        %Plot heatmap of the black image, smoothed and unsmoothed
        %PARAMETERS:
            %percentile: two vector, defines the limits of the greyvalues in the heatmap
        function plotBlackSurface(this,percentile)
            this.imagesc_truncate(this.black_train,percentile);
            colorbar;
            this.imagesc_truncate(this.black_fit,percentile);
            colorbar;
        end
        
        %PLOT WHITE SURFACE
        %Plot heatmap of the white image, smoothed and unsmoothed
        %PARAMETERS:
            %percentile: two vector, defines the limits of the greyvalues in the heatmap
        function plotWhiteSurface(this,percentile)
            this.imagesc_truncate(this.white_train,percentile);
            colorbar;
            this.imagesc_truncate(this.white_fit,percentile);
            colorbar;
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
        %Get image of fitted image subtract a bw image, selected
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
        %Fit the train_index-th image from this.bw_stack
        %for all parameters in parameter_array.
        %For each parameter, calculate the mean squared error
        %PARAMETERS:
            %train_index: the image for training
            %parameter_array: vector of parameters
        %RETURN:
            %mse_array: row vector containing the mse for each parameter
        function mse_array = crossValidation(this,train_index,parameter_array)
            %get the number of parameters
            n_parameter = numel(parameter_array);
            %declare array of mse for each parameter
            mse_array = zeros(1,n_parameter);
            %for each parameter
            for i_parameter = 1:n_parameter
                %fit the surface
                this.fitSurface(train_index,parameter_array(i_parameter));
                %declare array of mse for each test image
                mse_sample = zeros(1,this.n_bw-1);
                %declare variable for counting the number of test images gone through so far - 1
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
                mse_array(i_parameter) = mean(mse_sample);
            end
            
        end
        
        %ROTATE CROSS VALIDATION
        %Do cross validation using each image as a training set once
        %PARAMETER:
            %parameter_array: array of parameters to consider
        %RETURN:
            %mse_array: this.bw x numel(parameter_array) matrix containing the mse
        function mse_array = rotateCrossValidation(this,parameter_array)
            %declare array of mse for each parameter
            mse_array = zeros(this.n_bw,numel(parameter_array));
            %for each image
            for i_data_index = 1:this.n_bw
                disp(i_data_index);
                %get the mse for each order
                mse_array(i_data_index,:) = this.crossValidation(i_data_index,parameter_array);
            end         
        end
        
        %GET Z RESIDUAL
        %For each black/white image, fit surface and get the residuals.
        %For each pixel, a z_residual is the average residual / (std(residual)/this.n_bw)
        %PARAMETER:
            %parameter: parameter of the surface to fit
        %RETURN:
            %z_residual: matrix (active_size size) of the z_residuals
        function z_residual = getZResidual(this,parameter)
            %declare array of this.n_bw matrices of this.active_size size
            residual = zeros([this.active_size,this.n_bw]);
            %for each b/w image
            for i_data_index = 1:this.n_bw
                %fit the surface
                this.fitSurface(i_data_index,parameter);
                %get the residual and save it in array
                residual(:,:,i_data_index) = this.getResidualImage(i_data_index);
            end
            %get the mean residual over all this.n_bw images
            mean_residual = mean(residual,3);
            %get the sum of squared residuals
            sxx_residual = sqrt(sum((residual-repmat(mean_residual,1,1,this.n_bw)).^2,3));
            %work out z_residual
            z_residual = this.n_bw * mean_residual ./ sxx_residual;
        end
        
        %PLOT SCAN
        %Plot heatmap of a scan image
        %PARAMETERS:
            %index: plot the index-th image
            %percentile: two vector, defines the limits of the greyvalues in the heatmap
        function plotScan(this,index,percentile)
            %get the index-th scan
            scan = this.scan_stack(:,:,index);
            this.imagesc_truncate(scan,percentile);
        end
        
        %PLOT SCAN SHADED CORRECTED USING FITTED SURFACE
        function plotScanShadeCorrect_fit(this,index,percentile)
            %shade correct the scan
            scan = this.shadeCorrect_fit(this.scan_stack(:,:,index));
            this.imagesc_truncate(scan,percentile);
        end
        
        %PLOT SCAN SHADED CORRECTED USING TRAINING IMAGES
        function plotScanShadeCorrect_train(this,index,percentile)
            %shade correct the scan
            scan = this.shadeCorrect_train(this.scan_stack(:,:,index));
            this.imagesc_truncate(scan,percentile);
        end
        
        %PLOT WHITE SHADED CORRECTED
        function plotWhiteShadeCorrect(this,percentile)
            this.imagesc_truncate(this.shadeCorrect_fit(this.white_train),percentile);
        end
        
        %PLOT BLACK SHADED CORRECTED
        function plotBlackShadeCorrect(this,percentile)
            this.imagesc_truncate(this.shadeCorrect_fit(this.black_train),percentile);
        end
        
        %FUNCTION: SHADE CORRECT USING FITTED SURFACE
        %PARAMETER:
            %image: matrix of greyvalues
        %RETURN:
            %image: shade corrected image using the fitted surface
        function image = shadeCorrect_fit(this,image)
            image = (image - this.black_fit) ./ (this.white_fit - this.black_fit);
        end
        
        %FUNCTION: SHADE CORRECT USING TRAINING IMAGE
        %PARAMETER:
            %image: matrix of greyvalues
        %RETURN:
            %image: shade corrected image using the fitted image
        function image = shadeCorrect_train(this,image)
            image = (image - this.black_train) ./ (this.white_train - this.black_train);
        end
        
    end
    
    methods (Static)
        
        %IMAGE SCALE TRUNCATE
        %Plot heatmap of image to scale using a percentile of the data
        %PARAMETERS:
            %image: matrix of greyvalues
            %percentile: two vector of percentages
        function imagesc_truncate(image,percentile)
            clims = prctile(reshape(image,[],1),percentile);
            figure;
            imagesc(image,clims);
            colormap gray;
        end
        
    end
    
    methods (Abstract)
        %FIT PANEL
        %Fit surface to a panel
        %PARAMETERS:
            %x: vector of x coordinates
            %y: vector of y coordinates
            %z: vector of greyvalues
            %p: parameter
        %RETURN:
            %sfit_obj: sfit object (surface fit)
        fitPanel(this,x,y,z,p);
        
    end
    
end

