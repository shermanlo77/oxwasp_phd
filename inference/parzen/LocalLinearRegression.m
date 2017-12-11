%CLASS: LOCAL LINEAR REGRESSION
%A class used for local linear regression
%
%Pass the data x and y through the constructor
%Regress using the method y_0 = getRegression(x_0)
%Getting the kernel parameter can be done using the method setLambda(lambda)
classdef LocalLinearRegression < handle
    
    %MEMBER VARAIBLES
    properties (SetAccess = private)
        
        x_data; %column vector of independent variables
        y_data; %column vector for dependent variables
        n; %number of data points
        lambda; %kernel parameter
        
        %normalisation constant
        %x_data and y_data will store data which has mean 0 and std 1
        %the orginial mean and std will be stored in the below 4 member variables
        y_shift; %normalisation constant for the center location of y
        y_scale; %normalisation constant for the scale of y
        x_shift; %normalisation constant for the center location of x
        x_scale; %normalisation constant for the scale of x
        
    end
    
    %METHODS
    methods (Access = public)
        
        %CONSTRUCTOR
        %PARAMETERS:
            %x: vector of independent variables
            %y: vector of dependent variables
        function this = LocalLinearRegression(x,y)
            
            %get rid of nan
            is_nan = isnan(x) | isnan(y);
            x = x(~is_nan);
            y = y(~is_nan);
            
            %get the scale and location normalisation constants
            %store it in the member variables
            this.x_shift = mean(x);
            this.x_scale = std(x);
            this.y_shift = mean(y);
            this.y_scale = std(y);
            
            %normalise the data to have mean 0 and std 1
            %store the normalised data
            this.x_data = (x-this.x_shift) / this.x_scale;
            this.y_data = (y-this.y_shift) / this.y_scale;
            
            %get the number of data
            this.n = numel(x);
            
            %get the kernel parameter
            this.lambda = 0.5;
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
        function [y_0, error] = getRegression(this,x_0)
            
            %normalise x_0
            x_0 = (x_0 - this.x_shift) / this.x_scale;
            
            %get kernel weights
            w = this.getKernel(x_0);
            %get the data with positive weights
            x_sub = this.x_data(w>0);
            y_sub = this.y_data(w>0);
            w = w(w>0);
            %get the number of data with positive weights
            n_sub = numel(w);
            
            %normalise the weights so that the sum = n
            w = (n_sub/sum(w)) * w;
            
            %get the square root of the weights
            w_root = sqrt(w);
            
            %get the weighted design matrix
            X = [ones(n_sub,1),x_sub];
            X(:,1) = X(:,1) .* w_root;
            X(:,2) = X(:,2) .* w_root;
            
            %get the weighted response vector
            y_sub = w_root .* y_sub;
            
            %estimate the regression parameter using QR
            [Q,R] = qr(X);
            beta_estimate = R \ (Q'*y_sub);
            
            %use the beta estimate to do local linear regression
            x_0 = [1;x_0];
            y_0 = x_0' * beta_estimate;
            %un-normalise y_0
            y_0 = (y_0*this.y_scale) + this.y_shift;
            
            %if the error of y_0 is requested, work it out
            if nargout >= 2
                %get the weighted RSS divided by n degrees of freedom
                var_estimate = mean( w.*((y_sub - X*beta_estimate).^2)) / (n_sub-2);
                %get the matrix WX, that is diag(w) * X
                WX = X;
                WX(:,1) = X(:,1) .* w_root;
                WX(:,2) = X(:,2) .* w_root;
                %estimate the covariance matrix of beta
                beta_cov = var_estimate * (X'*WX);
                %estimate the variance of y_0
                error = sqrt(x_0' * beta_cov * x_0);
                %get the std of y_0
                error = error * this.y_scale;
            end
            
%             %OLD CODE, estimating error in optima value of the minima
%             %this is done by estimating the variance of alpha/beta
%             optima_error = beta_cov(1)/(beta_estimate(2)^2) + beta_cov(end)*(beta_estimate(1)/(beta_estimate(2)^2))^2 - 2*beta_cov(2)*beta_estimate(1)/(beta_estimate(2)^3);
%             optima_error = sqrt(optima_error);
%             optima_error = optima_error * this.x_scale;
            
        end
        
        %METHOD: GET KERNEL
        %Return the kernel evaluated at x_0
        %PARAMETERS:
            %x_0: where the kernel shall be evaluated
        %RETURN:
            %k: kernel at x_0
        function k = getKernel(this, x_0)
            %Epanechnikov kernel
            z = (this.x_data - x_0) / this.lambda;
            k = zeros(this.n,1);
            k(abs(z)>=1) = 0;
            k(abs(z)<1) = 0.75 * (1-z(abs(z)<1).^2);
        end
        
    end
    
end

