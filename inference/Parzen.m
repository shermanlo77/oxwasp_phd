%CLASS: PARZEN ESTIMATION FOR DENSITY
%Class for estimating the density given data
%
%The method used is to take an average of the Gaussian kernel over all data
%
%See: Friedman, J., Hastie, T. and Tibshirani, R., 2001. The elements of statistical learning. New York: Springer series in statistics.
%
classdef Parzen < handle
    
    %MEMBER VARIABLE
    properties (SetAccess = protected)
        parameter; %parameter for the parzen, here it is the gaussian std
        data; %column vector of data
        n_data; %size of the data;
    end
    
    %METHODS
    methods (Access = public)
        
        %CONSTRUCTOR
        %PARAMETERS:
            %parameter: gaussian std
        function this = Parzen(parameter)
            this.parameter = parameter;
        end
        
        %METHOD: ADD DATA
        %PARAMETERS:
            %data: column vector of data
        function addData(this, data)
            this.data = data;
            this.n_data = numel(data);
        end
        
        %METHOD: GET DENSITY ESTIMATE
        %PARAMETERS:
            %x: column vector of values in the support
        function p = getDensityEstimate(this, x)
            %get the number of evaluations to do
            n = numel(x);
            %assign an array with the same size as x
            p = x;
            %for each value in x
            for i_x = 1:n
                %find the difference between x(i_x) and all entries in this.data
                d = x(i_x) - this.data;
                %take the sum of the Gaussian kernels
                p(i_x) = sum(normpdf(d/this.parameter))/(this.n_data * this.parameter);
            end            
        end
        
    end
    
end

