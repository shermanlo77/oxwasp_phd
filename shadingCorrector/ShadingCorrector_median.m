classdef ShadingCorrector_median < ShadingCorrector_smooth
    %ShadingCorrector_mean Shading corrected which smooths the
    %reference images using k nearest median
    
    properties
    end
    
    %METHODS
    methods
        
        %CONSTRUCTOR
        %Call superclass
        function this = ShadingCorrector_median(reference_image_array)
            this = this@ShadingCorrector_smooth(reference_image_array);
        end
        
        %SMOOTH PANEL
        %Smooth a panel of an image in this.reference_image_array
        %PARAMETERS:
            %index: pointer to an image in this.reference_image_array
            %corner_position: 2x2 matrix, 1st column represent the top left
            %of the panel, 2nd column represent the bottom right of the
            %panel, all inclusive
            %p: odd integer, size of the square to do mean smoothing
        function smoothPanel(this,index,corner_position,p)
            
            %get vector range from the corner parameter
            height_range = corner_position(1,1) : corner_position(1,2);
            width_range = corner_position(2,1) : corner_position(2,2);
            
            %get panel image
            panel_image = this.orginial_reference_array(height_range,width_range,index);
            %smooth the panel image
            panel_image = medfilt2(panel_image,[p,p],'symmetric');            
            %update the reference image with the smoothed panel
            this.reference_image_array(height_range,width_range,index) = panel_image;
            
        end
        
    end
    
end

