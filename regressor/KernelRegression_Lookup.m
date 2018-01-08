%CLASS: KERNEL REGRESSION LOOKUP
%See superclass KernelRegression
%
%The lookup version saves the prediction for each integer ranging from the min and max of the training set feature
%Prediction is then done using the saved predictions rather than regressing
%
%Key:
%   ### : may be implemented differently in other languages
classdef KernelRegression_Lookup < KernelRegression
    
    %MEMBER VARIABLES
    properties (SetAccess = private)
        x_range;
        y_lookup;
        regressor;
    end
    
    %METHODS
    methods (Access = public)
        
        %CONSTRUCTOR
        %PARAMETERS:
            %kernel: Kernel object, it must have the method evaluate()
            %scale_parameter: scale parameter to scale the data when evaluating the kernel
        function this = KernelRegression_Lookup(kernel, scale_parameter)
            %call superclass
            this@KernelRegression(kernel, scale_parameter);
            %instantise an object using the superclass ###
            this.regressor = KernelRegression(kernel, scale_parameter);
        end
        
        %OVERRIDE: TRAIN
        %Pass a training set, saves the prediction for each integer which spans x
        %PARAMETERS:
            %x: column vector of features
            %y: column vector of responses
        function train(this, x, y)
            %train the kernel regressor
            this.regressor.train(x,y); % ### superclass could of been used
            %get each integer which spans x
            this.x_range = floor(min(x)) : ceil(max(x));
            %instantise an array which covers each interger
            this.y_lookup = zeros(size(this.x_range));
            %for each integer, save the predicted response given that integer feature
            for i_x = 1:numel(this.x_range)
                this.y_lookup(i_x) = this.regressor.predict(this.x_range(i_x)); %###rather use this.predict@KernelRegression
            end
        end
        
        %OVERRIDE: PREDICT
        %Predict the response, given a feature
        %The feature is rounded and then the lookup table is used
        %PARAMETERS:
            %x: column vector, where to evaluate the kernel regression
        %RETURN:
            %y: column vector the evaluation of the kernel regression at point x
        function y = predict(this, x)
            %declare array the same size as x
            y = zeros(size(x));
            %for each point in x
            for i_x = 1:numel(x)
                %round x
                x_int = round(x(i_x));
                %check the boundary of x
                if x_int < this.x_range(1)
                    x_int = this.x_range(1);
                elseif x_int > this.x_range(end)
                    x_int = this.x_range(end);
                end
                %use the lookup table to predict
                y(i_x) = this.y_lookup(x_int - this.x_range(1) + 1);
            end
        end
        
    end
    
end

