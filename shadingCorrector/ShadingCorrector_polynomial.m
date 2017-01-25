classdef ShadingCorrector_polynomial < ShadingCorrector_smooth
    %ShadingCorrector_polynomial Shading corrected which smooths the
    %reference images with polynomial surfaces
    
    properties
    end
    
    %METHODS
    methods
        
        %CONSTRUCTOR
        %Call superclass
        function this = ShadingCorrector_polynomial(reference_image_array)
            this = this@ShadingCorrector_smooth(reference_image_array);
        end
        
        %SMOOTH PANEL
        %Smooth a panel of an image in this.reference_image_array
        %PARAMETERS:
            %index: pointer to an image in this.reference_image_array
            %corner_position: 2x2 matrix, 1st column represent the top left
            %of the panel, 2nd column represent the bottom right of the
            %panel, all inclusive
            %p: order of the polynomial surface
        function smoothPanel(this,index,corner_position,p)
            
            %get vector range from the corner parameter
            height_range = corner_position(1,1) : corner_position(1,2);
            width_range = corner_position(2,1) : corner_position(2,2);
            
            %get vector of greyvalues
            z = reshape(this.orginial_reference_array(height_range,width_range,index),[],1);
            %normalise the greyvalues to have mean 0 and std 1
            shift = mean(z);
            scale = std(z);
            z = (z-shift)/scale;
            
            %obtain the range of x and y in grid form
            [x_grid,y_grid] = meshgrid(1:numel(width_range),1:numel(height_range));
            %convert x,y,z from grid form to vector form
            x = reshape(x_grid,[],1);
            y = reshape(y_grid,[],1);
            
            %fit polynomial
            polynomial_string = strcat('poly',num2str(p),num2str(p));
            sfit_obj = fit([x,y],z,polynomial_string);
            
            %update the reference image with the polynomial fit
            this.reference_image_array(height_range,width_range,index) = sfit_obj(x_grid,y_grid)*scale + shift;
            
        end
        
    end
    
end

