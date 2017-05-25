classdef PanelPolynomialFitter < PolynomialFitter
    %PANELPOLYNOMIALFITTER 
    %   Fit panel-wise 2nd order polynonimal to the image. The coordinates
    %   of each panel should be obtained from the member variable
    %   data_information, this is assigned via the constructor.
    %   See POLYNOMIALFITTER for inherited methods
    
    %MEMBER VARIABLES
    properties
        %object containing information about the panels
        panel_counter
    end
    
    %METHODS
    methods
        
        %CONSTRUCTOR
        %PARAMETERS:
            %data_information: object containing information about the panels
        function this = PanelPolynomialFitter(panel_counter)
            %call super class
            this = this@PolynomialFitter();
            %assign member variables
            this.panel_counter = panel_counter;
        end
        
        %METHOD: FIT POLYNOMIAL
        %Fit a global 2nd order polynomial to the image
        %The fitted image is save to the member variable fitted_image
        function fitPolynomial(this, image)
            %get the dimensions of the image
            [height, width] = size(image);
            %assign a blank image to the member variable fitted_image
            this.fitted_image = zeros(height, width);
            
            %reset the counter for the panels
            this.panel_counter.resetPanelCorner();
            %for each panel
            while this.panel_counter.hasNextPanelCorner()
                %get the coordinate of the panels
                    %dim 1: [y coordinate, x coordinate]
                    %dim 2: [top left, bottom right]
                corner = this.panel_counter.getNextPanelCorner();
                %get the two vector containing the coordinates of the top left and bottom right of the panel
                %in the format of [y coordinate, x coordinate]
                top_left = corner(:,1);
                bottom_right = corner(:,2);
                
                %extract the panel from the image
                panel_image = image(top_left(1):bottom_right(1), top_left(2):bottom_right(2));
                
                %fit polynomial to that panel
                polynomial_fitter = PolynomialFitter();
                polynomial_fitter.fitPolynomial(panel_image);
                %put the fitted polynomial to the fitted image corresponding panel
                this.fitted_image(top_left(1):bottom_right(1), top_left(2):bottom_right(2)) = polynomial_fitter.fitted_image;
             
            end %while

        end %fitPolynomial
        
    end %method
    
end

