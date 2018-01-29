classdef QuadraticRegression < LinearRegression

    properties
    end
    
    methods
        
        function this = QuadraticRegression()
            this@LinearRegression();
            this.n_feature = 3;
        end
        
        %GET DESIGN MATRIX
        %Returns a design matrix with polynomial features given a column vector of explanatory variables
        %PARAMETERS:
            %x: column vector of greyvalues
        %RETURN:
            %X: n x this.n_order + 1 design matrix
        function X = getDesignMatrix(this,x)
            %declare design matrix
            X = zeros(numel(x),this.n_feature);
            %first column is a constant
            X(:,1) = 1;
            X(:,2) = x;
            X(:,3) = x.^2;
        end
        
    end
    
end

