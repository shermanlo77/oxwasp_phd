%MIT License
%Copyright (c) 2019 Sherman Lo

%CLASS: PARZEN ESTIMATION FOR DENSITY
%Class for estimating the density given data
%The method used is to take an average of the Gaussian kernel over all data
%See: Friedman, J., Hastie, T. and Tibshirani, R., 2001. The elements of statistical learning. 
    %New York: Springer series in statistics.
classdef Parzen < handle
  
  %MEMBER VARIABLE
  properties (SetAccess = protected)
    bandwidth; %bandwidth for the parzen, here it is the gaussian std
    data; %column vector of data
    nData; %size of the data;
  end
  
  %METHODS
  methods (Access = public)
    
    %CONSTRUCTOR
    %PARAMETERS:
      %data: column vector of data
    function this = Parzen(data)
      this.data = data(~isnan(data));
      this.nData = numel(this.data);
      %set default value for parzen std
      this.setParameter( (0.9*this.nData^(-1/5)+0.16) * min([std(this.data),iqr(this.data)/1.34]));
    end
    
    %METHOD: SET PARAMETER
    %Set the parzen std
    %PARAMETERS:
    %bandwidth: parzen std
    function setParameter(this, bandwidth)
      this.bandwidth = bandwidth;
    end
    
    %METHOD: GET DENSITY ESTIMATE
    %PARAMETERS:
      %x: column vector of values in the support
    %RETURN:
      %p: density for each entry in x
    function p = getDensityEstimate(this, x)
      %get the number of evaluations to do
      n = numel(x);
      %assign an array with the same size as x
      p = x;
      %for each value in x
      for i_x = 1:n
        %evaluate the estimated density
        p(i_x) = this.getSumKernel(x(i_x))/(this.nData * this.bandwidth);
      end
    end
    
  end
  
  methods (Access = private)
    
    %METHOD: GET SUM OF KERNELS
    %Return the sum of Gaussian kernels at each data point
    %PARAMETER:
      %x: center of the Gaussian kernel
    %RETURN:
      %s: sum of Gaussian kernels at each data point
    function s = getSumKernel(this, x)
      %find the difference between x(i_x) and all entries in this.data
      d = x - this.data;
      %take the sum of the Gaussian kernels
      s = sum(normpdf(d/this.bandwidth));
    end
    
  end
  
end
