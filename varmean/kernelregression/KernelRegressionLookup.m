%MIT License
%Copyright (c) 2019 Sherman Lo

%CLASS: KERNEL REGRESSION LOOKUP
%See superclass KernelRegression
%
%The lookup version saves the prediction for each integer ranging from the min and max of the
%training set feature
%Prediction is then done using the saved predictions rather than regressing
classdef KernelRegressionLookup < handle
  
  %MEMBER VARIABLES
  properties (SetAccess = private)
    kernel;
    bandwidth;
    xRange;
    yLookup;
  end
  
  %METHODS
  methods (Access = public)
    
    %CONSTRUCTOR
    %PARAMETERS:
      %kernel: Kernel object, it must have the method evaluate()
      %bandwidth: scale parameter to scale the data when evaluating the kernel
    function this = KernelRegressionLookup(kernel, bandwidth)
      this.kernel = kernel;
      this.bandwidth = bandwidth;
    end
    
    %OVERRIDE: TRAIN
    %Pass a training set, saves the prediction for each integer which spans x
    %PARAMETERS:
      %x: column vector of features
      %y: column vector of responses
    function train(this, x, y)
      %train the kernel regressor
      kernelRegression = KernelRegression(this.kernel, this.bandwidth);
      kernelRegression.train(x,y);
      %get each integer which spans x
      this.xRange = (floor(min(x)) : ceil(max(x)))';
      %instantise an array which covers each interger
      this.yLookup = zeros(size(this.xRange));
      %for each integer, save the predicted response given that integer feature
      for i = 1:numel(this.xRange)
        this.yLookup(i) = kernelRegression.predict(this.xRange(i));
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
      x = int32(round(x));
      x(x<this.xRange(1)) = this.xRange(1);
      x(x>this.xRange(2)) = this.xRange(2);
      y = this.yLookup(x - this.xRange(1) + 1);
    end
    
  end
  
end
