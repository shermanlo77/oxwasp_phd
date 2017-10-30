%CLASS: GENERALISED BOX PLOT
%Redefine the outlier to consider heavy tailed and skewed data
classdef BoxplotGeneralised < Boxplot
    
    %MEMBER VARIABLES
    properties (SetAccess = private)
        turkey_quantile; %the quantile used for turkey parameters estimation
        alpha; %end of whisker coverage of the distribution, that is 1-2*alpha is the coverage of the whisker
        shift; %used to set the mimimum value of the data when translating it to be all positive
        n_monte_carlo; %number of monte carlo simulations to get the inverse cdf of the turkey distribution
    end
    
    methods (Access = public)
        
        %CONSTRUCTOR
        function this = BoxplotGeneralised(X)
            %call superclass constructor
            this@Boxplot(X);
            %assign member variables
            this.turkey_quantile = 0.1;
            this.alpha = 1 - normcdf(4*norminv(0.75));
            this.shift = 0.1;
            this.n_monte_carlo = 1E4;
        end
        
    end
    
    methods (Access = protected)
       
        %OVERRIDE: GET OUTLIER
        %Set which data are outliers or not, save the boolean in the member variable outlier_index
        %Uses the method described in Bruffaerts (2014)
        function getOutlier(this)
            
            %shift the data to be all positive, mimimum is this.shift
            r = this.X - min(this.X) + this.shift;
            %normalise it, perserving rank
            r = r/(min(r)+max(r));
            
            %transform it using the inverse cdf
            w = norminv(r);
            %normalise it to have std 1
            w = 1.3426 * (w-median(w)) / iqr(w);

            %get the quantiles of w
            q = quantile(w,[this.turkey_quantile, 1-this.turkey_quantile]);
            %get the quantiles of the Normal distribution
            z_p = norminv(1-this.turkey_quantile,0,1);

            %estimate the parameters of the Turkey distribution
            g = log(-q(2)/q(1))/z_p;
            h = 2*log(-g * (q(2)*q(1)) / (sum(q))) / (z_p^2);

            %set a deterministic random number generator
            stream = RandStream('mt19937ar','Seed',0);
            %simulate Normal random variabkes
            Z = stream.randn(this.n_monte_carlo,1);
            %transform the Normal random variables to Turkey random varibales
            Y = (exp(g*Z)-1).*exp(h*Z.^2/2) / g;

            %find the critical values for w
            %this is found by using the quantiles of the imperical Turkey density
            w_critical = quantile(Y,[this.alpha, 1-this.alpha]);

            %for each outlier, set the boolean in this.outlier_index ot be true
            this.outlier_index = (w < w_critical(1)) | (w > w_critical(2));
        end
        
    end
    
end

