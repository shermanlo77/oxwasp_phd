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
            this.density_estimation_parameter = 0;
            
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
            %calculate the p values
            this.getPValues();
            %instantise p tester and do test
            p_tester = PTester(this.p_image, this.size);
            p_tester.doTest();
            %save the results, sig_image and size_corrected
            this.sig_image = p_tester.sig_image; %2d boolean of significant pixels
            this.size_corrected = p_tester.size_corrected; %size of test, corrected for multiple testing
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
            %instantise a parzen density estimator
            density_estimator = Parzen(reshape(this.z_image(~isnan(this.z_image)),[],1));
            %if the density estimation parameter is not 0, change the parameter of the parzen density estimator
            if this.density_estimation_parameter ~= 0
                density_estimator.setParameter(this.density_estimation_parameter);
            end
        end
        
        %METHOD: ESTIMATE NULL RATIO
        %Estimates the proportion of data which are null using the peak densities
        %RETURN:
            %p0: proportion of data which are null
        function p0 = estimateP0(this)
           p0 =  this.density_estimator.getDensityEstimate(this.mean_null)/normpdf(0,0,this.std_null);
           p0 = min([p0,1]);
        end
        
        %METHOD: ESTIMATE LOCAL FDR
        %Estimates the fdr using the densities
        %RETURN:
            %fdr: local false discovery rate
        function fdr = estimateLocalFdr(this, x)
            fdr = this.estimateP0() * normpdf(x,this.mean_null,this.std_null) ./ this.density_estimator.getDensityEstimate(x);
        end
        
        %METHOD: ESTIMATE TAIL FDR (two tailed)
        %Estimates the fdr using cdf
        %RETURN:
            %fdr: tail false discovery rate
        function fdr = estimateTailFdr(this, x)
            %indicate z values which are less than the mean
            is_left = x < this.mean_null;
            
            %declare an array left tail and right tail, all with initial value 0
            %left tail contain values at the left tail to be evaluated using the cdf
            %right tail contain values at the right tail to be evaluated using the cdf
            left_tail = x;
            right_tail = x;
            left_tail(:) = 0;
            right_tail(:) = 0;
            
            %for values less than the mean, copy it over to the left tail array
            left_tail(is_left) = x(is_left);
            %for values more than the mean, copy it over to the right tail array
            right_tail(~is_left) = x(~is_left);
            
            %for values more than the mean, reflect it to the left tail and save it to the left tail array
            left_tail(~is_left) = 2*this.mean_null - x(~is_left);
            %for values less than the mean, reflect it to the right tail and save it to the right tail array
            right_tail(is_left) = 2*this.mean_null - x(is_left);
            
            %get the evaluation of the null cdf at the left tail and the right tail, add them together
            F0 = normcdf(left_tail,this.mean_null,this.std_null) + normcdf(right_tail,this.mean_null,this.std_null,'upper');
            %get the evaluation of the non-null cdf at the left tail and the right tail, add them together
            F = this.density_estimator.getCdfEstimate(left_tail,false) + this.density_estimator.getCdfEstimate(right_tail,true);
            
            %estimate the tail fdr
            fdr = this.estimateP0() * F0 ./ F;
            
        end
        
        %METHOD: GET Q IMAGE
        %Return the q image, this is the tail fdr evaluated for each point in z image
        %RETURN
            %q_image: image of q values
        function q_image = getQImage(this)
            %reshape the z values into a column vector
            q_image = reshape(this.z_image,[],1);
            %estimate the tail fdr, evaluated for each z value
            q_image = this.estimateTailFdr(q_image);
            %reshape the z values into an image
            q_image = reshape(q_image,size(this.z_image));
        end
        
        %METHOD: GET Q CRITICAL
        %Return the critical q values, this is the tail fdr evaluated at z critical
        function q_critical = getQCritical(this)
            %evaluate the tail fdr at the z critical
            q_critical = this.estimateTailFdr(this.getZCritical());
            %both the left and right critical boundary should produce the same q value
            %take the 1st one
            q_critical = q_critical(1);
        end
        
        %METHOD: ESTIMATE H1 DENSITY
        %Estimates the alternate densisty
        %PARAMETERS:
            %x: points to be evaluated by the alternative density
        %RETURN:
            %f1: alternate density evaluated at x
        function f1 = estimateH1Density(this, x)
            %estimate the alternative density
            f1 = (this.density_estimator.getDensityEstimate(x) - this.estimateP0()*normpdf(x,this.mean_null,this.std_null)) / (1-this.estimateP0());
            %ensure all values of the density at non-negative
            f1(f1<0) = 0;
        end
        
        %METHOD: ESTIMATE H1 CDF
        %Estimates the alternative cdf
        %PARAMETERS:
            %x: points to be evaluated by the alternative cdf
            %is_upper: boolean, true if to evaluate the right hand side of the cdf
        %RETURN:
            %F1: alternative cdf evaluated at x
        function F1 = estimateH1Cdf(this, x, is_upper)
            %evaluate the cdf using the corresponding tail
            if is_upper
                F0 = normcdf(x,this.mean_null,this.std_null,'upper');
            else
                F0 = normcdf(x,this.mean_null,this.std_null);
            end
            %estimate the H1 cdf
            F1 = (this.density_estimator.getCdfEstimate(x, is_upper) - this.estimateP0()*F0) / (1-this.estimateP0());
        end
        
        %METHOD: ESTIMATE POWER
        %Estimate the power using the estimated H1 and the critical boundary
        function power = estimatePower(this)
            %get the critical boundaries
            z_critical = this.getZCritical();
            %estimate the power
            power = this.estimateH1Cdf(z_critical(1), false) + this.estimateH1Cdf(z_critical(2), true);
        end
        
    end
    
end

