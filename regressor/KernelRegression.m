%CLASS: KERNEL REGRESSION
%Does kernel regression given a kernel, scale parameter and a training set
%Pass the kernel and scale parameter through the contructor
%Pass the training data using the method train()
%Get predictions using the method predict()
classdef KernelRegression < Regressor
    
    %MEMBER VARIABLES
    properties (SetAccess = private)
        kernel;
        scale_parameter;
        x_train;
        y_train;
    end
    
    %METHODS
    methods (Access = public)
        
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
            %x: column vector, where to evaluate the kernel regression
        %RETURN:
            %y: column vector the evaluation of the kernel regression at point x
        function y = predict(this, x)
            %declare a column vector of predictions
            y = zeros(size(x));
            %for each point in x
            for i = 1:numel(x)
                %get the distance to x and divide it by scale_parameter
                z = (x(i) - this.x_train) / this.scale_parameter;
                %evaluate the kernel for each value in x
                k = this.kernel.evaluate(z);
                %do kernel regression, this is a weighted sum
                y(i)= sum(k.*this.y_train) / sum(k);
            end
        end
        
        %GET NAME
        %Return name for this glm
        function name = getName(this)
            name = cell2mat({this.kernel.name,'-',num2str(this.scale_parameter)});
        end
        
        %GET FILE NAME
        function name = getFileName(this)
            name = cell2mat({this.kernel.name,'_',num2str(this.scale_parameter)});
        end
        
    end
    
end

