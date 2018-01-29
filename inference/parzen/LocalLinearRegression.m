%CLASS: LOCAL LINEAR REGRESSION
%A class used for local linear regression
%
%Pass the data x and y through the constructor
%Regress using the method y_0 = getRegression(x_0)
%Getting the kernel parameter can be done using the method setLambda(lambda)
classdef LocalLinearRegression < Regressor
    
    %MEMBER VARAIBLES
    properties (SetAccess = private)
        
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

                linear_regression = LinearRegression();
                linear_regression.addWeights(w);
                linear_regression.train(this.x_data,this.y_data);
                y(i) = linear_regression.predict(x_0);

    %             %if the error of y_0 is requested, work it out
    %             if nargout >= 2
    %                 %get the weighted RSS divided by n degrees of freedom
    %                 var_estimate = mean( w.*((y_sub - X*beta_estimate).^2)) / (n_sub-2);
    %                 %get the matrix WX, that is diag(w) * X
    %                 WX = X;
    %                 WX(:,1) = X(:,1) .* w_root;
    %                 WX(:,2) = X(:,2) .* w_root;
    %                 %estimate the covariance matrix of beta
    %                 beta_cov = var_estimate * (X'*WX);
    %                 %estimate the variance of y_0
    %                 error = sqrt(x_0' * beta_cov * x_0);
    %                 %get the std of y_0
    %                 error = error * this.y_scale;
    %             end

    %             %OLD CODE, estimating error in optima value of the minima
    %             %this is done by estimating the variance of alpha/beta
    %             optima_error = beta_cov(1)/(beta_estimate(2)^2) + beta_cov(end)*(beta_estimate(1)/(beta_estimate(2)^2))^2 - 2*beta_cov(2)*beta_estimate(1)/(beta_estimate(2)^3);
    %             optima_error = sqrt(optima_error);
    %             optima_error = optima_error * this.x_scale;
            end
            
        end
        
        
        function has_errorbar = hasErrorbar(this)
            has_errorbar = false;
        end
        
    end
    
end

