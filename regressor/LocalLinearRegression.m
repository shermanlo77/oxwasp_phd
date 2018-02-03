%CLASS: LOCAL LINEAR REGRESSION
%A class used for local linear regression
%
%Pass the kernel through the constructor
%Call the method train to train the regression
%Regress using the method y_0 = getRegression(x_0)
classdef LocalLinearRegression < Regressor
    
    %MEMBER VARAIBLES
    properties (SetAccess = protected)
        x_data; %column vector of feature variables
        y_data; %column vector for response variables
        n; %number of data points
        lambda; %kernel parameter
        kernel; %kernel object
    end
    
    %METHODS
    methods (Access = public)
        
        %CONSTRUCTOR
        %PARAMETERS:
            %kernel: kernel object
            %lambda: parameter for the kernel
        function this = LocalLinearRegression(kernel, lambda)
            %assign member variables
            this.kernel = kernel;
            this.lambda = lambda;
        end
        
        %METHOD: TRAIN
        %Train the local linear regression
        %PARAMETERS:
            %x: column vector of feature variables
            %y: column vector for response variables
        function train(this,x,y)
            %get rid of nan
            is_nan = isnan(x) | isnan(y);
            %assign the training set
            this.x_data = x(~is_nan);
            this.y_data = y(~is_nan);
        end
        
        %METHOD: GET REGRESSION
        %Return the evaluation of the local linear regression at x_0
        %PARAMETERS:
            %x_0: vector, where the evaluation of the local linear regression takes place
        %RETURN:
            %y_0: vector, evaluation of the local linear regression at x_0
        function y = predict(this,x)
            %declare array of y or predictions
            y = zeros(size(x));
            %for each element in x
            for i = 1:numel(x)
                %get the ith element
                x_0 = x(i);
                %get kernel weights for each data in the training set
                w = this.kernel.evaluate((x_0 - this.x_data)/this.lambda);
                %get the regression
                local_regression = this.getLocalRegression();
                %add the kernel weight
                local_regression.addWeights(w);
                %train the regression
                local_regression.train(this.x_data,this.y_data);
                %get the prediction of the regression
                y(i) = local_regression.predict(x_0);
            end
        end
        
        %IMPLEMENTED: HAS ERROR BAR
        function has_errorbar = hasErrorbar(this)
            has_errorbar = false;
        end
        
    end
    
    %PROTECTED METHOD
    methods (Access = protected)
        
        %METHOD: GET LOCAL REGRESSION
        %Returns a regression object for the local regression
        %RETURN:
            %regression: regression object for local regression
        function regression = getLocalRegression(this)
            regression = LinearRegression();
        end
        
    end
    
end

