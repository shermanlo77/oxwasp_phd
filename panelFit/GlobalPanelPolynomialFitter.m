classdef GlobalPanelPolynomialFitter < PolynomialFitter
    %GLOBALPANELPOLYNOMIALFITTER 
    %   Fit panel-wise and a global 2nd order polynonimal to the image.
    %   The coordinates of each panel should be obtained from the member variable
    %   data_information, this is assigned via the constructor.
    %   See POLYNOMIALFITTER for inherited methods
    
    %MEMBER VARIABLES
    properties
        %object containing information about the panels
        data_information
    end
    
    %METHODS
    methods
        
        %CONSTRUCTOR
        function this = GlobalPanelPolynomialFitter(data_information)
            %call super class
            this = this@PolynomialFitter();
            %assign member variables
            this.data_information = data_information;
        end
        
        %METHOD GET DESIGN MATRIX
        %PARAMETERS:
            %width: width of the image
            %height: height of the image
        %RETURN:
            %design matrix of size width*height x 197 matrix
            %dim 1: for each pixel in a width x height image
            %dim 2: 2nd order polynomial features (x, y, x*y, x^2, y^2)
                %then for i = 1,2,...,number of panels
                %append by (constant, x, y, x*y, x^2, y^2) * indicator ((x,y) is in panel i)
                %
                %alternative explanation
                %(x, y, x*y, x^2, y^2)
                %then (constant, x, y, x*y, x^2, y^2) for panel 1
                %then (constant, x, y, x*y, x^2, y^2) for panel 2
                %...
                %then (constant, x, y, x*y, x^2, y^2) for the last panel
        function X = getDesignMatrix(this, width, height)
            
            %declare the design matrix
            X = zeros(width*height, 5+6*this.data_information.n_panel);
            %get the feature vector of the regular polynomial
            X(:,1:5) = PolynomialFitter.getPolynomialDesignMatrix(width, height);
            
            %reset the counter for the panels
            this.data_information.resetPanelCorner();
            %for each panel
            panel_count = 1; %count the number of panels on the go
            intercept = 1; %value for the intercept
            
            %for each panel
            while this.data_information.hasNextPanelCorner()
                %get the coordinate of the panels
                    %dim 1: [y coordinate, x coordinate]
                    %dim 2: [top left, bottom right]
                corner = this.data_information.getNextPanelCorner();
                %get the two vector containing the coordinates of the top left and bottom right of the panel
                %in the format of [y coordinate, x coordinate]
                top_left = corner(:,1);
                bottom_right = corner(:,2);
                
                %get the area of the panel
                panel_area = prod(bottom_right - top_left + 1);
                %declare a vector, an element for each pixel in the panel
                %this will contain that pixel's corresponding row number in the design matrix
                index_vector = zeros(panel_area, 1);
                
                %count the number of pixels on the go
                index_count = 1;
                
                %for each pixel in the panel
                for x = top_left(2) : bottom_right(2)
                    for y = top_left(1) : bottom_right(1)
                        %get the index of that pixel
                        index_vector(index_count) = (x-1)*height + y;
                        %increment the pixel count
                        index_count = index_count + 1;
                    end
                end
                
                %assign the intercept feature for this particular panel to the design matrix
                X(index_vector,5+((panel_count-1)*6+1)) = intercept;
                %get the columns of the design matrix, which should contain polynomial features for this particular panel
                column_index = (5+((panel_count-1)*6+2)) : (5+((panel_count-1)*6+6));
                %assign polynomial features for this particular panel to the design matrix
                X(index_vector,column_index) = X(index_vector,1:5);
                
                %increment the panel count
                panel_count = panel_count + 1;
                
            end %while
            
        end %constructor
        
    end %method
    
end %class

