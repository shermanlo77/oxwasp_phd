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
        var_null; %variance of the null hypothesis
        size; %size of the test, default is 2 sigma OR 2*normcdf(-2)
        size_corrected; %corrected size of the test due to multiple testing
        p0; %estimation of propotion of H0 data, default is 1 for the sake of scaling without emperical null correction
        
        p_image; %2d array of p_values
        sig_image; %boolean 2d array, true if that pixel is significant
        n_test; %number of tests
        density_estimator; %density estimator object
        
        n_step; %maximum number of steps in the newton raphson method
        tol; %the smallest change of the grad density to declare convergence in the newton raphson method
        quantile_array; %quantiles to use for the initial value for newton raphson
        rng; %random number generator for when newton raphson method fails
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
            this.var_null = 1;
            this.p0 = 1;
            this.setSigma(2);
            
            %get the number of non_nan values in z_image
            nan_index = isnan(reshape(z_image,[],1));
            this.n_test = sum(~nan_index);
            
            this.density_estimator = Parzen(reshape(this.z_image(~isnan(this.z_image)),[],1));
            
            this.n_step = 10;
            this.tol = 1E-5;
            this.quantile_array = [0.25,0.5,0.75];
            this.rng = RandStream('mt19937ar','Seed',507957022);
        end
        
        %METHOD: SET SIZE
        %Set the size of the hypothesis test
        %PARAMETERS:
            %size: the size of the hypothesis test
        function setSize(this, size)
            this.size = size;
        end
        
        %METHOD: SET SIGMA
        %Set the size of the hypothesis test to be 2*normcdf(-sigma)
        %PARAMETERS:
            %sigma: threshold of the test
        function setSigma(this, sigma)
            this.setSize(2*normcdf(-sigma));
        end
        
        %METHOD: SET DENSITY ESTIMATION PARAMETER
        %Set the std of the gaussian kernel used in density estimation
        %PARAMETERS:
            %density_estimation_parameter: std of the gaussian kernel used in density estimation
        function setDensityEstimationParameter(this,density_estimation_parameter)
            this.density_estimator.setParameter(density_estimation_parameter);
        end
        
        %METHOD: SET DENSITY ESTIMATION FUDGE FACTOR
        %Set the std of the gaussian kernel used in density estimation by using a fudge factor
        %parzen std = gradient x std x n^(-1/5) + intercept
        %PARAMETERS:
            %gradient
            %intercept
        function setDensityEstimationFudgeFactor(this,gradient,intercept)
            this.density_estimator.setFudgeFactor(gradient,intercept);
        end
        
        %METHOD: ESTIMATE NULL
        %Estimates the mean and var null hypothesis using a fitted density
        %The mean is found using the Newton-Raphson method, starting point at the mean
        %The std is found using the second derivate of the log density at the mode
        %Updates the parameters of the null hypothesis, stored in the member variables
        function estimateNull(this, ~)
            %if the density estimator has no data, return nan
            if this.density_estimator.n_data == 0
                this.mean_null = nan;
                this.var_null = nan;
            %else get the emperical null parameters
            else
                
                %declare array of starting initial values
                z_initial_array = quantile(this.density_estimator.data,this.quantile_array);
                %declare array to store mode solutions, lnf and d2x_lnf for each of the modes
                z_null_array = zeros(numel(this.quantile_array),1);
                lnf_array = zeros(numel(this.quantile_array),1);
                d2x_lnf_array = zeros(numel(this.quantile_array),1);
                
                %for each starting point
                for i_quantile = 1:numel(this.quantile_array)
                    %get the mode, log density and d2x log density
                    [z_null_array(i_quantile), lnf_array(i_quantile), d2x_lnf_array(i_quantile)] = this.findMode(z_initial_array(i_quantile));
                end
                
                %get the z with the maximum log density
                [lnf_max,z_index] = max(lnf_array);
                %if the log density is nan, repeat
                if isnan(lnf_max)
                    %warn for using random initial values
                    warning('using random initial values');
                    %set a boolean flag for convergence
                    has_converge = false;
                    %while the mode has not been found, ie Newton-Raphson has not converged yet
                    while ~has_converge
                        z_null = quantile(this.density_estimator.data,this.rng.rand());
                        [z_null, ~, d2x_lnf] = this.findMode(z_null);
                        %check if the solution to the mode is a maxima by looking at the 2nd diff
                        if d2x_lnf < 0
                            %then the algorithm has converged
                            has_converge = true;
                        end
                    end
                    %save the mode
                    this.mean_null = z_null;
                %else, save the mode and 2nd derivate
                else
                    this.mean_null = z_null_array(z_index);
                    d2x_lnf = d2x_lnf_array(z_index);
                end
                
                %estimate the null var
                this.var_null = -1/d2x_lnf;
                %check if the std_null is real
                if ~isreal(this.var_null)
                    this.var_null = nan;
                end

                %estimate p0, propotion of H0 data
                this.p0 =  this.density_estimator.getDensityEstimate(this.mean_null)/normpdf(0,0,sqrt(this.var_null));
                this.p0 = min([this.p0,1]);
                
            end
            
        end
        
        %METHOD: FIND MODE
        %Find the mode of the density estimator using the Newton Raphson algorithm
        %PARAMETER:
            %z_null: initial value
        %RETURN:
            %z_null: mode
            %lnf: log density at the mode
            %d2x_lnf: 2nd derivate of the log density at the mode
        function [z_null, lnf, d2x_lnf] = findMode(this, z_null)
            %get the 1st and 2nd diff of the ln density at the initial value
            [dx_lnf, d2x_lnf] = this.density_estimator.getDLnDensity(z_null);
            %for n_step
            for i_step = 1:this.n_step
                %update the solution to the mode
                z_null = z_null - dx_lnf/d2x_lnf;
                %get the 1st and 2nd diff of the ln density at the new value
                [dx_lnf, d2x_lnf] = this.density_estimator.getDLnDensity(z_null);
                %if this gradient is within tolerance, break the i_step for loop
                if (abs(dx_lnf)<this.tol)
                    break;
                end
                %if any of the variables are nan, break the loop as well
                if any(isnan([dx_lnf, d2x_lnf, z_null]))
                    break;
                end
            end
            %check if the solution to the mode is a maxima by looking at the 2nd diff
            %return the log density
            if d2x_lnf < 0
                lnf = log(this.density_estimator.getDensityEstimate(z_null));
            else
                z_null = nan;
                lnf = nan;
            end
        end
        
        %METHOD: GET Z CORRECTED
        %Returns the z image, corrected under the emperical null hypothesis
        %RETURNS:
            %z_corrected: 2d array of z statistics, corrected under the emperical null hypothesis
        function z_corrected = getZCorrected(this)
            z_corrected = (this.z_image - this.mean_null) / sqrt(this.var_null);
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
            z_critical = z_critical*sqrt(this.var_null) + this.mean_null;
        end

        %METHOD: ESTIMATE H1 DENSITY
        %Estimates the alternate densisty
        %PARAMETERS:
            %x: points to be evaluated by the alternative density
        %RETURN:
            %f1: alternate density evaluated at x
        function f1 = estimateH1Density(this, x)
            %estimate the alternative density
            f1 = (this.density_estimator.getDensityEstimate(x) - this.p0*normpdf(x,this.mean_null,sqrt(this.var_null))) / (1-this.p0);
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
                F0 = normcdf(x,this.mean_null,sqrt(this.var_null),'upper');
            else
                F0 = normcdf(x,this.mean_null,sqrt(this.var_null));
            end
            %estimate the H1 cdf
            F1 = (this.density_estimator.getCdfEstimate(x, is_upper) - this.p0*F0) / (1-this.p0);
        end
        
        %METHOD: ESTIMATE POWER
        %Estimate the power using the estimated H1 and the critical boundary
        function power = estimatePower(this)
            %get the critical boundaries
            z_critical = this.getZCritical();
            %estimate the power
            power = this.estimateH1Cdf(z_critical(1), false) + this.estimateH1Cdf(z_critical(2), true);
        end
        
        %METHOD: ESTIMATE ALTERNATIVE
        %Assume the alternative distribution is Gaussian
        %Estimate the parameters of the alternative distribution using the mode and curvature at the mode
        %PARAMETERS:
            %linspace: number of points between the min and max used for mode seeking
        %RETURN:
            %mean_h1: mean of H1
            %std_h1: std of H1
        function [mean_h1, std_h1] = estimateAlternative(this, n_linspace)
            %declare n_linspace equally spaced points between min and max of z_array
            z_array = linspace(min(min(this.z_image)), max(max(this.z_image)), n_linspace);
            %get the alternative density evaluated at each point in z_array
            f1 = this.estimateH1Density(z_array);
            
            %find the index with the biggest density
            [~,z_index] = max(f1);
            %get the mode, the value of z with the biggest density
            mean_h1 = z_array(z_index);
            
            %get the null density and its derivate
            f0 = this.density_estimator.getDensityEstimate(mean_h1);
            f0_d1 = f0 * (mean_h1 - this.mean_null)/sqrt(this.var_null);
            f0_d2 = f0 * ( (mean_h1 - this.mean_null)^2/this.var_null + 1) / this.var_null;
            
            %get the non-null density and its derivate
            f1 = this.estimateH1Density(mean_h1) * this.p0;
            f1_d1 = this.density_estimator.getDensity_d1(mean_h1) - this.p0*f0_d1;
            f1_d2 = this.density_estimator.getDensity_d2(mean_h1) - this.p0*f0_d2;
            
            %work out std_h1 using the curvature at the mode
            d2 = (f1*f1_d2 - (f1_d1)^2)/(f1^2);
            std_h1 = (-d2)^(-1/2);
            
        end
        
        %METHOD: FIGURE HISTOGRAM CRITICAL BOUNDARY
        %Produce a figure plots:
            %histogram of z statistics
            %emperical null
            %critical boundary
        function fig = figureHistCritical(this)
            %produce a figure
            fig = figure;
            %plot histogram
            this.plotHistogram();
            hold on;
            ax = gca;
            %get 500 values from min to max
            z_plot = linspace(ax.XLim(1),ax.XLim(2),500);
            %plot null
            this.plotNull(z_plot);
            %plot critical boundary
            this.plotCritical();
            %label axis and legend
            xlabel('z statistic');
            ylabel('frequency density');
            legend('histogram','null','critical boundary');
        end
        
        %METHOD: FIGURE HISTOGRAM DENSITY CRITICAL BOUNDARY
        %Produce a figure plots:
            %histogram of z statistics
            %density estimate
            %emperical null
            %emperical alt
            %critical boundary
        function fig = figureHistDensityCritical(this)
            %produce a figure
            fig = figure;
            %plot histogram
            this.plotHistogram();
            hold on;
            %plot critical boundary
            this.plotCritical();
            
            ax = gca;
            %get 500 values from min to max
            z_plot = linspace(ax.XLim(1),ax.XLim(2),500);
            %plot density estimate
            this.plotDensityEstimate(z_plot);
            %plot null density
            this.plotNull(z_plot);
            %plot alt density
            this.plotAlt(z_plot);
            
            %label axis and legend
            xlabel('z statistic');
            ylabel('frequency density');
            legend(ax.Children([6,3,2,1,5]),'histogram','estimated density','null','alt','critical boundary');
        end
        
        %METHOD: PLOT HISTOGRAM
        %Plot histogram of z statistics
        function plotHistogram(this)
            z_vector = reshape(this.z_image,[],1);
            z_vector(isnan(z_vector)) = [];
            histogram_custom(z_vector);
        end
        
        %METHOD: PLOT CRITICAL BOUNDARY
        function plotCritical(this)
            ax = gca;
            plot([norminv(this.size_corrected/2,this.mean_null,sqrt(this.var_null)),norminv(this.size_corrected/2,this.mean_null,sqrt(this.var_null))],[0,ax.YLim(2)],'r--');
            hold on;
            plot([norminv(1-this.size_corrected/2,this.mean_null,sqrt(this.var_null)),norminv(1-this.size_corrected/2,this.mean_null,sqrt(this.var_null))],[0,ax.YLim(2)],'r--');
        end
        
        %METHOD: PLOT DENSITY ESTIMATE
        %PARAMETERS:
            %z_plot: values to evaluate the density estimate at
        function plotDensityEstimate(this, z_plot)
            plot(z_plot,this.density_estimator.getDensityEstimate(z_plot)*this.n_test);
        end
        
        %METHOD: PLOT NULL DENSITY
        %PARAMETERS:
            %z_plot: values to evalute the null density at
        function plotNull(this, z_plot)
            plot(z_plot,normpdf(z_plot,this.mean_null,sqrt(this.var_null))*this.n_test*this.p0);
        end
        
        %METHOD: PLOT ALT DENSITY
        %PARAMETERS:
            %z_plot: values to evalute the alt density at
        function plotAlt(this, z_plot)
            plot(z_plot,(1-this.p0)*this.n_test*this.estimateH1Density(z_plot));
        end
        
    end
    
end

%         %METHOD: ESTIMATE LOCAL FDR
%         %Estimates the fdr using the densities
%         %RETURN:
%             %fdr: local false discovery rate
%         function fdr = estimateLocalFdr(this, x)
%             fdr = this.p0 * normpdf(x,this.mean_null,sqrt(this.var_null)) ./ this.density_estimator.getDensityEstimate(x);
%         end
%         
%         %METHOD: ESTIMATE TAIL FDR (two tailed)
%         %Estimates the fdr using cdf
%         %RETURN:
%             %fdr: tail false discovery rate
%         function fdr = estimateTailFdr(this, x)
%             %indicate z values which are less than the mean
%             is_left = x < this.mean_null;
%             
%             %declare an array left tail and right tail, all with initial value 0
%             %left tail contain values at the left tail to be evaluated using the cdf
%             %right tail contain values at the right tail to be evaluated using the cdf
%             left_tail = x;
%             right_tail = x;
%             left_tail(:) = 0;
%             right_tail(:) = 0;
%             
%             %for values less than the mean, copy it over to the left tail array
%             left_tail(is_left) = x(is_left);
%             %for values more than the mean, copy it over to the right tail array
%             right_tail(~is_left) = x(~is_left);
%             
%             %for values more than the mean, reflect it to the left tail and save it to the left tail array
%             left_tail(~is_left) = 2*this.mean_null - x(~is_left);
%             %for values less than the mean, reflect it to the right tail and save it to the right tail array
%             right_tail(is_left) = 2*this.mean_null - x(is_left);
%             
%             %get the evaluation of the null cdf at the left tail and the right tail, add them together
%             F0 = normcdf(left_tail,this.mean_null,sqrt(this.var_null)) + normcdf(right_tail,this.mean_null,sqrt(this.var_null),'upper');
%             %get the evaluation of the non-null cdf at the left tail and the right tail, add them together
%             F = this.density_estimator.getCdfEstimate(left_tail,false) + this.density_estimator.getCdfEstimate(right_tail,true);
%             
%             %estimate the tail fdr
%             fdr = this.p0 * F0 ./ F;
%             
%         end
%         
%         %METHOD: GET Q IMAGE
%         %Return the q image, this is the tail fdr evaluated for each point in z image
%         %RETURN
%             %q_image: image of q values
%         function q_image = getQImage(this)
%             %reshape the z values into a column vector
%             q_image = reshape(this.z_image,[],1);
%             %estimate the tail fdr, evaluated for each z value
%             q_image = this.estimateTailFdr(q_image);
%             %reshape the z values into an image
%             q_image = reshape(q_image,size(this.z_image));
%         end
%         
%         %METHOD: GET Q CRITICAL
%         %Return the critical q values, this is the tail fdr evaluated at z critical
%         function q_critical = getQCritical(this)
%             %evaluate the tail fdr at the z critical
%             q_critical = this.estimateTailFdr(this.getZCritical());
%             %both the left and right critical boundary should produce the same q value
%             %take the 1st one
%             q_critical = q_critical(1);
%         end
%         
%         %METHOD: ESTIMATE POWER (USING EXP TAIL FDR)
%         %Estimate the power using the average tail fdr under the non-null density
%         %PARAMETERS:
%             %a: starting point for the trapezium rule
%             %b: end point for the trapezium rule
%             %n: number of trapeziums
%         %RETURN:
%             %power: statistical power
%         function power = estimateTailPower(this, a, b, n)
%             power = this.estimateFdrPower(true, a, b, n);
%         end
%         
%         %METHOD: ESTIMATE POWER (USING EXP LOCAL FDR)
%         %Estimate the power using the average local fdr under the non-null density
%         %PARAMETERS:
%             %a: starting point for the trapezium rule
%             %b: end point for the trapezium rule
%             %n: number of trapeziums
%         %RETURN:
%             %power: statistical power
%         function power = estimateLocalPower(this, a, b, n)
%             power = this.estimateFdrPower(false, a, b, n);
%         end
%         
%         %METHOD: ESTIMATE POWER (USING EXP FDR)
%         %Estimate the power using the average fdr under the non-null density
%         %PARAMETERS:
%             %is_tail: boolea, true if to use the tail fdr, else use local fdr
%             %a: starting point for the trapezium rule
%             %b: end point for the trapezium rule
%             %n: number of trapeziums
%         %RETURN:
%             %power: statistical power
%         function power = estimateFdrPower(this, is_tail, a, b, n)
%             %get n equally spaced points, from a to b
%             x = linspace(a,b,n);
%             %get the height of the trapeziums
%             h = (b-a)/n;
%             %get the density estimate of the non-null density
%             f1 = this.estimateH1Density(x);
%             %get the estimated fdr
%             if is_tail
%                 fdr = this.estimateTailFdr(x);
%             else
%                 fdr = this.estimateLocalFdr(x);
%             end
%             %get the integrand
%             I = f1.*fdr;
%             
%             %integrate I, divide by the normalisation constand
%             power = (0.5*h*(I(1)+I(end)+2*sum(I(2:(end-1))))) / (0.5*h*(f1(1)+f1(end)+2*sum(f1(2:(end-1)))));
%             %get the power
%             power = 1 - power;
%         end

