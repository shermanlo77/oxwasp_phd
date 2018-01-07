%CLASS: KERNEL REGRESSION
%Does kernel regression given a kernel, scale parameter and a training set
%Pass the kernel and scale parameter through the contructor
%Pass the training data using the method train()
%Get predictions using the method predict()
classdef KernelRegression < handle
    
    %MEMBER VARIABLES
    properties (SetAccess = private)
        kernel;
        scale_parameter;
        x_train;
        y_train;
    end
    
    %METHODS
    methods
        
        %CONSTRUCTOR
        %PARAMETERS:
            %kernel: Kernel object, it must have the method evaluate()
            %scale_parameter: scale parameter to scale the data when evaluating the kernel
        function this = KernelRegression(kernel, scale_parameter)
            %assign member variables
            this.kernel = kernel;
            this.scale_parameter = scale_parameter;
        end
        
        %METHOD: TRAIN
        %Pass a training set
        %PARAMETERS:
            %x: column vector of features
            %y: column vector of responses
        function train(this, x, y)
            %assign member variables
            this.x_train = x;
            this.y_train = y;
        end
        
        %METHOD: PREDICT
        %Do kernel regression at the point x
        %PARAMETERS:
            %x: where to evaluate the kernel regression
        %RETURN:
            %y: the evaluation of the kernel regression at point x
        function y = predict(this, x)
            %for each point in x_train, get the distance to x and divide it by scale_parameter
            z = (x - this.x_train) / this.scale_parameter;
            %evaluate the kernel for each value in x
            k = this.kernel.evaluate(z);
            %do kernel regression, this is a weighted sum
            y = sum(k.*this.y_train) / sum(k);
        end
        
    end
    
end

