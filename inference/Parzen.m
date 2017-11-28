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
            %data: column vector of data
        function this = Parzen(data)
            this.data = data(~isnan(data));
            this.n_data = numel(this.data);
            %set default value for parzen std using Silverman's rule of thumb
            %this.setParameter(1.144 * min([std(this.data),iqr(this.data)/1.34]) * this.n_data^(-1/5) );
            this.setParameter(3.33 * min([std(this.data),iqr(this.data)/1.34]) * this.n_data^(-1/5) );
            %this.setParameter(0.4);
        end
        
        %METHOD: SET PARAMETER
        %Set the parzen std
        %PARAMETERS:
            %parameter: parzen std
        function setParameter(this, parameter)
            this.parameter = parameter;
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
                p(i_x) = this.getSumKernel(x(i_x))/(this.n_data * this.parameter);
            end            
        end
        
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
            s = sum(normpdf(d/this.parameter));
        end
        
        %METHOD: GET CDF ESTIMATE
        %PARAMETERS:
            %x: column vector of values in the support
            %is_upper: boolean, true if want right tail cdf
        %RETURN:
            %p: column vector of percentiles for each entry in x
        function p = getCdfEstimate(this,x,is_upper)
            %get the number of evaluations to do
            n = numel(x);
            %assign an array with the same size as x
            p = x;
            %for each value in x
            for i_x = 1:n
                %find the difference between x(i_x) and all entries in this.data
                d = x(i_x) - this.data;
                %take the sum of the Gaussian kernels using the corresponding tailed cdf
                if is_upper
                    p(i_x) = sum(normcdf(d/this.parameter,'upper'))/this.n_data;
                else
                    p(i_x) = sum(normcdf(d/this.parameter))/this.n_data;
                end
            end            
        end
        
        %METHOD: GET KOLMOGOROV-SMIRNOV STATISTIC
        %Compares the fitted cdf with the data in x_test
        %PARAMETERS:
            %x_test: column vector of data to compare with the fitted cdf
        %RETURN:
            %D: KS statistic
        function D = getKSStatistic(this, x_test)
            %set the size of x_test
            n_test = numel(x_test);
            %get the cdf evaluated at each entry in x_test
            p_fitted = this.getCdfEstimate(x_test);
            %get the emperical cdf (aka percentile) at each entry in x_test
            p_histogram = ( ((1:n_test) - 0.5) / n_test )';
            %work out the KS statistic
            D = max(abs(p_fitted - p_histogram));
        end
        
        %METHOD: TUNE AND SET PARAMETER
        %Gets the KS statistic for each parameter in parameter_array using k-fold cross validation
        %Sets the parameter with the best KS statistic
        %PARAMETERS:
            %k: number of folds
            %parameter_array: column vector of parameters to try out in k-fold cross validation
        function ks_array = tuneSetParameter(this,k,parameter_array)
            %set up k fold cross validation
            k_folder = KFolder(k, this.n_data);
            %get the number of parameters in parameter_array
            n_parameter = numel(parameter_array);
            %declare array for storing ks statistics for each fold and parameter
            ks_array = zeros(k,n_parameter);
            %for each parameter
            for i_parameter = 1:n_parameter
                %for each fold
                for i_fold = 1:k
                    %get the index of the training set and test set
                    training_index = k_folder.getTrainingSet();
                    test_index = k_folder.getTestSet();
                    %instantise a parzen with the training data
                    parzen_i = Parzen(this.data(training_index));
                    %set that parzen to have this current parameter
                    parzen_i.setParameter(parameter_array(i_parameter));
                    %get the KS statistic comparing the test data with the fitted cdf
                    ks_array(i_fold, i_parameter) = parzen_i.getKSStatistic(this.data(test_index));
                    %rotate the folds
                    k_folder.rotateFolds();
               end
            end
            %get the index of the min ks statistic
            [~,index_max] = min(mean(ks_array));
            %set this to have that parameter, the one with the minimum ks statistic
            this.setParameter(parameter_array(index_max));
        end
        
        %METHOD: GET LOG SECOND DERIVATE
        %Return the second derivate of the log density at the value delta
        %PARAMETER:
            %delta: the value at which the second derivate of the log density to be evaluated at
        function d2 = getLogSecondDerivate(this, delta)
            %get the Gaussian kernel parameter
            z = (this.data - delta)/this.parameter;
            %work out the second derivate of the log density
            d2 = (this.getSumKernel(delta) * sum(normpdf(z).*(z.^2-1)) - (sum(normpdf(z).*z))^2) / (this.parameter*this.getSumKernel(delta))^2;
        end
        
        %METHOD: ESTIMATE NULL
        %Estimates the parameters of the null distribution
        %The mean is found using the maximum value of the fitted density on n_linspace equally spaced points between min and max of z_array
        %The std is found using the second derivate of the log density at the mode
        %PARAMETER:
            %n_linspace: number of equally spaced points between min and max of z_array to find the mode of the estimated density
        %RETURN:
            %mean_null: expectation of the null hypothesis
            %std_null: std of the null hypothesis
        function [mean_null, std_null] = estimateNull(this, n_linspace)
            
            %there must be more than 1 data point to contine, otherwise return NaN
            if this.n_data >= 2
            
                %declare n_linspace equally spaced points between min and max of z_array
                z_array = linspace(min(this.data), max(this.data), n_linspace);
                %get the density estimate at each point
                density_estimate = this.getDensityEstimate(z_array);
                %find the index with the highest density
                [~, z_max_index] = max(density_estimate);

                %the z with the highest density is the mode
                mean_null = z_array(z_max_index);
                %estimate the null std using the log second derivate
                std_null = (-this.getLogSecondDerivate(mean_null))^(-1/2);
                %if the calculation of the estimator of std_null is complex
                %set it to nan
                if ~isreal(std_null)
                   std_null = nan; 
                end
            %there must be more than 1 data point to contine, otherwise return NaN
            else
                mean_null = nan;
                std_null = nan;
            end
        end
        
    end
    
end

