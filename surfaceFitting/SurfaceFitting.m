classdef SurfaceFitting < handle
    %SURFACEFITTING Summary of this class goes here
    %   Detailed explanation goes here
    
    %MEMBER VARIABLES
    properties
        
        total_size; %two vector [height, width] representing the total size of the detector image in pixels
        active_size; %two vector [height, width] representing the total size of the cropped image in pixels
        n_panel_column; %scalar representing the number of columns of 2 panels in the detector
        
        panel_height; %height of each panel in pixels
        panel_width; %width of each non-border panel in pixels
        panel_width_edge; %width of panels on the edges (far left and far right) in pixels
        
        black_stack; %array of black images
        black_train; %a black image
        black_fit; %smoothed black image
        
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
            this.n_panel_column = n_panel_column;
            
            %panel height (there are 2 rows of panels)
            this.panel_height = this.active_size(1)/2;
            %panel width (divide the width by the number of columns of panels)
            this.panel_width = this.total_size(2)/n_panel_column;
            %panel width at the edges (from cropping)
            this.panel_width_edge = this.panel_width - ((this.total_size(2)-this.active_size(2))/2);
            
            %declare empty matrices
            this.black_fit = zeros(active_size);
        end
        
        %LOAD BLACK IMAGES
        function loadBlack(this,file_location)
            this.black_stack = load_black(file_location);
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
        
        %FIT POLYNOMIAL PANEL (BLACK)
        %Using an image from the stack of black images, fit polynomial
        %surface to it.
        %PARAMETERS:
            %train_index: the image from this.black_stack to be used
            %p: order of the polynomial
        function fitPolynomialPanel_black(this,train_index,p)
            
            %get the black image from the stack
            this.black_train = this.black_stack(:,:,train_index);
            
            %for each column
            for i_column = 1:this.n_panel_column
                
                %for each row
                for i_row = 1:2
                    
                    %get the range of rows which covers a panel
                    height_range = (1 + (i_row-1)*this.panel_height) : (i_row*this.panel_height);
                    
                    %then get the range of columns which covers that panel
                    %special case if the column is on the boundary
                    if i_column == 1
                        width_range = 1:(this.panel_width_edge);
                    elseif i_column == this.n_panel_column
                        width_range = (this.active_size(2)-this.panel_width_edge+1):this.active_size(2);
                    %ordinary case:
                    else
                        width_range = (this.panel_width_edge + (i_column-2)*this.panel_width + 1) : (this.panel_width_edge + (i_column-1)*this.panel_width);
                    end
                    
                    %for the given panel, get the grey values
                    z_grid = this.black_train(height_range,width_range);
                    %obtain the range of x and y in grid form
                    [x_grid,y_grid] = meshgrid(width_range,height_range);
                    %convert x,y,z from grid form to vector form
                    x = reshape(x_grid,[],1);
                    y = reshape(y_grid,[],1);
                    z = reshape(z_grid,[],1);
                    
                    %fit polynomial surface to the data (x,y,z)
                    sfit_obj = this.fitPolynomialSurface(x,y,z,p);
                    
                    %save the surface of the panel to this.black_fit
                    this.black_fit(height_range,width_range) = sfit_obj(x_grid,y_grid);
                    
                end
                
            end
            
        end
        
        %CLEAR BLACK
        %Assign empty matrix to this.black_stack
        function clearBlack(this)
            this.black_stack = [];
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
            %heatmap plot the smooth data
            figure;
            imagesc(this.black_fit,clims);
        end
        
    end
    
end

