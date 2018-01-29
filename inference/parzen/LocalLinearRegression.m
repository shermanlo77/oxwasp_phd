%CLASS: LOCAL LINEAR REGRESSION
%A class used for local linear regression
%
%Pass the data x and y through the constructor
%Regress using the method y_0 = getRegression(x_0)
%Getting the kernel parameter can be done using the method setLambda(lambda)
classdef LocalLinearRegression < Regressor
    
    %MEMBER VARAIBLES
    properties (SetAccess = protected)
        
        x_data; %column vector of independent variables
        y_data; %column vector for dependent variables
        n; %number of data points
        lambda; %kernel parameter
        kernel;
        
    end
    
    %METHODS
    methods (Access = public)
        
        %CONSTRUCTOR
        %PARAMETERS:
            %x: vector of independent variables
            %y: vector of dependent variables
        function this = LocalLinearRegression(kernel, lambda)
            this.kernel = kernel;
            this.lambda = lambda;
        end
        
        function train(this,x,y)
            %get rid of nan
            is_nan = isnan(x) | isnan(y);
            this.x_data = x(~is_nan);
            this.y_data = y(~is_nan);
        end
        
        %METHOD: SET LAMBDA
        %Set the member variable lambda, a parameter for the kernel
        %PARAMETER:
            %lambda: kernel parameter
        function setLambda(this, lambda)
            this.lambda = lambda;
        end
        
        %METHOD: GET REGRESSION
        %Return the evaluation of the local linear regression at x_0
        %PARAMETERS:
            %x_0: where the evaluation of the local linear regression takes place
        %RETURN:
            %y_0: evaluation of the local linear regression at x_0
            %error: estimated error of y_0
        function [y] = predict(this,x)
            
            y = zeros(size(x));
            
            for i = 1:numel(x)
                x_0 = x(i);

                %get kernel weights
                w = this.kernel.evaluate((x_0 - this.x_data)/this.lambda);

                local_regression = this.getLocalRegression();
                local_regression.addWeights(w);
                local_regression.train(this.x_data,this.y_data);
                y(i) = local_regression.predict(x_0);
            end
            
        end
        
        function regression = getLocalRegression(this)
            regression = LinearRegression();
        end
        
        
        function has_errorbar = hasErrorbar(this)
            has_errorbar = false;
        end
        
    end
    
end

