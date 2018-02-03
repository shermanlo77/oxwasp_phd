%CLASS: QUADRATIC REGRESSION
%1D quadratic regression
classdef QuadraticRegression < LinearRegression

    %MEMBER VARIABLES
    properties
    end
    
    %METHODS
    methods (Access = public)
        
        %CONTRUCTOR
        function this = QuadraticRegression()
            %class superclass
            this@LinearRegression();
            %set the number of features to be 3
            this.n_feature = 3;
        end
        
        %OVERRIDE: GET DESIGN MATRIX
        %Returns a design matrix with polynomial features given a column vector of explanatory variables
        %PARAMETERS:
            %x: column vector of features
        %RETURN:
            %X: n x this.n_order + 1 design matrix
        function X = getDesignMatrix(this,x)
            %declare design matrix
            X = zeros(numel(x),this.n_feature);
            %first column is a constant
            X(:,1) = 1;
            %second column is linear
            X(:,2) = x;
            %third column is quadratic term
            X(:,3) = x.^2;
        end
        
    end
    
end

