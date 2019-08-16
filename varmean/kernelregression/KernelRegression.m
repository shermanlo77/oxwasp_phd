%MIT License
%Copyright (c) 2019 Sherman Lo

%CLASS: KERNEL REGRESSION
%Does kernel regression given a kernel, bandwidth and a training set
%Pass the kernel and scale parameter through the contructor
%Pass the training data using the method train()
%Get predictions using the method predict()
classdef KernelRegression < handle
  
  %MEMBER VARIABLES
  properties (SetAccess = private)
    nTrain;
    kernel;
    bandwidth;
    xTrain;
    yTrain;
  end
  
  %METHODS
  methods (Access = public)
    
    %CONSTRUCTOR
    %PARAMETERS:
      %kernel: Kernel object, it must have the method evaluate()
      %bandwidth: scale parameter to scale the data when evaluating the kernel
    function this = KernelRegression(kernel, bandwidth)
      %assign member variables
      this.kernel = kernel;
      this.bandwidth = bandwidth;
    end
    
    %METHOD: TRAIN
    %Pass a training set
    %PARAMETERS:
      %x: column vector of features
      %y: column vector of responses
    function train(this, x, y)
      %assign member variables
      this.xTrain = x;
      this.yTrain = y;
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
        z = (x(i) - this.xTrain) / this.bandwidth;
        %evaluate the kernel for each value in x
        k = this.kernel.evaluate(z);
        %do kernel regression, this is a weighted sum
        y(i)= sum(k.*this.yTrain) / sum(k);
      end
    end
    
  end
  
end
