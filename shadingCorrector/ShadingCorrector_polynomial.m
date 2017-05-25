classdef ShadingCorrector_polynomial < ShadingCorrector_smooth
    %ShadingCorrector_polynomial Shading corrected which smooths the
    %reference images with polynomial surfaces
    
    properties
    end
    
    %METHODS
    methods
        
        %CONSTRUCTOR
        %Call superclass
        function this = ShadingCorrector_polynomial(reference_image_array, panel_counter, parameter)
            this = this@ShadingCorrector_smooth(reference_image_array, panel_counter, parameter);
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
            
            %LEGACY CODE only support p=2 for now
            if p ~= 2
                warning('2nd order polynomial can only be supported');
            end
            
            %get vector range from the corner parameter
            height_range = corner_position(1,1) : corner_position(1,2);
            width_range = corner_position(2,1) : corner_position(2,2);
            
            %get the image of the panel
            image = this.orginial_reference_array(height_range,width_range,index);
            %fit a polynomial
            polynomial_fitter = PolynomialFitter();
            polynomial_fitter.fitPolynomial(image);
            %get the fitted polynomial
            image = polynomial_fitter.fitted_image;
            
            %update the reference image with the polynomial fit
            this.reference_image_array(height_range,width_range,index) = image;
            
        end
        
    end
    
end

