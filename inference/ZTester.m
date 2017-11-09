%CLASS Z TESTER
%Does hypothesis test on an image of z statistics
%Multiple testing corrected using FDR, see:
    %Benjamini, Y. and Hochberg, Y., 1995.
    %Controlling the false discovery rate: a practical and powerful approach to multiple testing.
    %Journal of the royal statistical society. Series B (Methodological), pp.289-300.
%The null hypothesis can be corrected using the empirical null using a density estimate
    %The density was estimated using Gaussian kernels (see Parzen class)
    %Also see:
    %Efron, B., 2004.
    %Large-scale simultaneous hypothesis testing: the choice of a null hypothesis.
    %Journal of the American Statistical Association, 99(465), pp.96-104.
classdef ZTester < handle

    %MEMBER VARIABLES
    properties (SetAccess = protected)
        z_image; %2d array of z statistics
        mean_null; %mean of the null hypothesis
        std_null; %std of the null hypothesis
        size; %size of the test, default is 2 sigma OR 2*normcdf(-2)
        size_corrected; %corrected size of the test due to multiple testing
        density_estimation_parameter; %std of the gaussian kernel used in density estimation
        p_image; %2d array of p_values
        sig_image; %boolean 2d array, true if that pixel is significant
        n_test; %number of tests
        density_estimator; %density estimator object
    end
    
    %METHODS
    methods (Access = public)
        
        %CONSTRUCTOR
        %PARAMETERS:
            %z_image: 2d array of z statistics
        function this = ZTester(z_image)
            %assign member variables
            this.z_image = z_image;
            %assign default values to the member variables
            this.mean_null = 0;
            this.std_null = 1;
            this.size = 2*normcdf(-2);
            this.density_estimation_parameter = 0.2;
            
            %get the number of non_nan values in z_image
            nan_index = isnan(reshape(z_image,[],1));
            this.n_test = sum(~nan_index);
        end
        
        %METHOD: SET SIZE
        %Set the size of the hypothesis test
        %PARAMETERS:
            %size: the size of the hypothesis test
        function setSize(this, size)
            this.size = size;
        end
        
        %METHOD: SET DENSITY ESTIMATION PARAMETER
        %Set the std of the gaussian kernel used in density estimation
        %PARAMETERS:
            %density_estimation_parameter: std of the gaussian kernel used in density estimation
        function setDensityEstimationParameter(this,density_estimation_parameter)
            this.density_estimation_parameter = density_estimation_parameter;
        end
        
        %METHOD: ESTIMATE NULL
        %Estimates the mean and std null hypothesis using a fitted density
        %The mean is found using the maximum value of the fitted density on n_linspace equally spaced points between min and max of z_array
        %The std is found using the second derivate of the log density at the mode
        %Updates the parameters of the null hypothesis, stored in the member variables
        %PARAMETERS:
            %n_linspace: number of equally spaced points between min and max of z_array to find the mode of the estimated density
        function estimateNull(this, n_linspace)
            %get the density estimator
            this.density_estimator = this.getDensityEstimator();
            %declare n_linspace equally spaced points between min and max of z_array
            z_array = linspace(min(min(this.z_image)), max(max(this.z_image)), n_linspace);
            %get the density estimate at each point
            density_estimate = this.density_estimator.getDensityEstimate(z_array);
            %find the index with the highest density
            [~, z_max_index] = max(density_estimate);
            
            %the z with the highest density is the mode
            this.mean_null = z_array(z_max_index);
            %estimate the null std using the log second derivate
            this.std_null = (-this.density_estimator.getLogSecondDerivate(this.mean_null))^(-1/2);   
        end
        
        %METHOD: GET Z CORRECTED
        %Returns the z image, corrected under the emperical null hypothesis
        %RETURNS:
            %z_corrected: 2d array of z statistics, corrected under the emperical null hypothesis
        function z_corrected = getZCorrected(this)
            z_corrected = (this.z_image - this.mean_null) / this.std_null;
        end
        
        %METHOD: GET P VALUES
        %Calculate the p values (corrected using the emperical null hypothesisis)
        function getPValues(this)
            %work out the p values
            this.p_image = 2*(normcdf(-abs(this.getZCorrected())));
        end
        
        %METHOD: DO TEST
        %Does hypothesis using the p values, corrected using FDR
        %Saves significant pixels in the member variable sig_image
        function doTest(this)
            
            %declare image of boolean values, true if that pixel is significant
            this.sig_image = this.z_image;
            this.sig_image(:) = false;

            %put the p values in a column vector
            p_array = reshape(this.p_image,[],1);
            %sort the p_array in accending order
            %p_ordered is p_array sorted
            %p_ordered_index contains indices of the values in p_ordered in relation to p_array
            [p_ordered, p_ordered_index] = sort(p_array);

            %find the index of p_ordered which is most significant using the FDR algorithm
            p_critical_index = find( p_ordered(~isnan(p_ordered)) <= this.size*(1:this.n_test)'/this.n_test, 1, 'last');

            %if there are p values which are significant
            if ~isempty(p_critical_index)

                %correct the size of the test using that p value
                this.size_corrected = p_ordered(p_critical_index);

                %set everything in p_array to be false
                %they will be set to true for significant p values
                p_array = zeros(numel(p_array),1);

                %using the entries indiciated by p_ordered_index from element 1 to p_critical_index
                %set these elements in sig_array to be true
                p_array(p_ordered_index(1:p_critical_index)) = true;

                %put p_array in non nan entries of sig_array
                this.sig_image(:) = p_array;
            else
                %correct the size of the test is the Bonferroni correction
                this.size_corrected = this.size / this.n_test;
            end
        end
        
        %METHOD: GET Z CIRITCAL
        %Return the critical values of z, corrected using the emperical null
        function z_critical = getZCritical(this)
            z_critical = norminv(this.size_corrected/2);
            z_critical(2) = -z_critical;
            z_critical = z_critical*this.std_null + this.mean_null;
        end
        
        %METHOD: GET DENSITY ESTIMATOR
        %Returns a density estimator object
        function density_estimator = getDensityEstimator(this)
            density_estimator = Parzen(reshape(this.z_image(~isnan(this.z_image)),[],1), this.density_estimation_parameter);
        end
        
    end
    
end

