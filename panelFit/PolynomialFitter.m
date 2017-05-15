classdef PolynomialFitter < handle
    %POLYNOMIALFITTER 
    %   Fit 2nd order polynomial to the image, this is done via the method
    %   fitPolynomial. The fitted image can be obtained via the member
    %   variable fitted_image.
    
    %MEMBER VARIABLES
    properties
        fitted_image; %image fitted with 2nd order polynomial
    end
    
    
    %METHODS
    methods
        
        %CONSTRUCTOR
        function this = PolynomialFitter()
        end
        
        %METHOD GET DESIGN MATRIX
        %See static class getPolynomialDesignMatrix
        %PARAMETERS:
            %width: width of the image
            %height: height of the image
        %RETURN:
            %width*height x 5 matrix
            %dim 1: for each pixel in a width x height image
            %dim 2: 2nd order polynomial features (x, y, x*y, x^2, y^2)
        function X = getDesignMatrix(this, width, height)
            X = PolynomialFitter.getPolynomialDesignMatrix(width, height);
        end %constructor
        
        %METHOD: FIT POLYNOMIAL
        %Fit a global 2nd order polynomial to the image
        %The fitted image is save to the member variable fitted_image
        function fitPolynomial(this, image)
            %Get the dimensions of the image
            [height, width] = size(image);
            
            %reshape the image into a vector, this is the response vector
            Y = reshape(image, [], 1);
            %normalise the response vector
            Y_mean = mean(Y);
            Y_std = std(Y);
            Y = (Y - Y_mean) / Y_std;
            
            %get the design matrix
            X = this.getDesignMatrix(width, height);

            %do least squares regression
            beta = X\Y;
            
            %plot the fitted 2nd order polynomial and save it to the member
            %variable fitted_image
            Y_fit = (X*beta)*Y_std + Y_mean;
            this.fitted_image = reshape(Y_fit,height,width);

        end %fitPolynomial
        
    end %methods
    
    
    %STATIC METHODS
    methods (Static)
        
        %METHOD: GET POLYNOMIAL DESIGN MATRIX
            %Returns a design matrix containing 2nd order polynomial features
            %for all the coordinates in a width x height image
            %They are ordered in matrix order, that is for each column and
            %work downwards
        %PARAMETERS:
            %width: width of the image
            %height: height of the image
        %RETURN:
            %width*height x 5 matrix
            %dim 1: for each pixel in a width x height image
            %dim 2: 2nd order polynomial features (x, y, x*y, x^2, y^2)
        function X = getPolynomialDesignMatrix(width, height)
            %declare a design matrix
            X = zeros(width*height, 5);
            %for each pixel in the image
            for x = 1:width
                for y = 1:height
                    %get the index, this is the same as the row number of
                    %the design matrix
                    i_row = (x-1)*height + y;
                    %fill in the row of the design matrix with the 2nd
                    %order polynomial features
                    X(i_row,1) = x;
                    X(i_row,2) = y;
                    X(i_row,3) = x*y;
                    X(i_row,4) = x^2;
                    X(i_row,5) = y^2;
                end
            end
            
            %normalise the design matrix
            X = (X - mean(X)) ./ std(X);
            
        end %getPolynomialDesignMatrix
    end %static methods
    
end

