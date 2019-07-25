classdef CompoundPoisson < handle

    %MEMBER VARIABLES
    properties
        lambda; %poisson parameter
        alpha; %gamma shape parameter
        beta; %gamma rate parameter
        phi; %dispersion parameter
        p; %power index
        mu; %mean
        sigma; %standard deviation
        X; %array of observed compound poisson variables
        Y; %array of latent poisson variables
        Y_var; %array of latent poisson variances
        n; %number of data points
        n_compound_poisson_term; %maximum number of terms to be calculated in the compound poisson sum
        
        
        compound_poisson_sum_threshold; %negative number
        %if ln(compound poisson term / biggest compound poisson term) > compound_poisson_sum_threshold
        %then that term is considered for the compound poisson sum
        
        can_support_zero_mass; %boolean, ture if can support probability mass at 0
        
        name; %name of this object (for figure saving purposes);
    end

    %METHODS
    methods
        
        %CONSTRUCTOR
        %PARAMETERS
            %X: vector of data
        function this = CompoundPoisson()
            %assign member variables
            this.n_compound_poisson_term = 1E7;
            this.compound_poisson_sum_threshold = -37;
            this.can_support_zero_mass = true;
            this.name = 'cp';
        end
        
        %METHOD: ADD DATA
        %Assign the member variable X
        %PARAMETERS:
            %X: vector of compound poisson variables
        function addData(this,X)
            this.X = X;
            this.n = numel(X);
        end
        
        %METHOD: ADD LATENT DATA
        %Assign the member variable Y
        %PARAMETERS:
            %Y: vector of poisson latent variables
        function addLatentData(this,Y)
            this.Y = Y;
        end
        
        %METHOD: INITALISE EM ALGORITHM
        function initaliseEM(this)
            this.Y = zeros(numel(this.X),1);
            this.Y_var = zeros(numel(this.X),1);
        end
        
        %METHOD: SET PARAMETERS
        %Assign the member variables lambda, alpha and beta
        %PARMAETERS:
            %lambda: poisson parameter
            %alpha: gamma shape parameter
            %beta: gamma rate parameter
        function setParameters(this,lambda,alpha,beta)
            this.lambda = lambda;
            this.alpha = alpha;
            this.beta = beta;
            this.p = (alpha+2)/(alpha+1); 
            this.phi = (1+alpha)*beta^(this.p-2)*(alpha*lambda)^(1-this.p);
            this.mu = this.lambda * this.alpha / this.beta;
            this.sigma = sqrt(this.phi * this.mu^this.p);
        end
        
        %METHOD: GET DENSITY
        %PARAMETER:
            %x: vector of compound Poisson variables
        %RETURN:
            %pdf: vector of densities, for each element in x
        function pdf = getPdf(this,x)
            pdf = zeros(numel(x),1);
            for i = 1:numel(x)
                pdf(i) = exp(this.getlnpdf(x(i)));
            end
        end
        
        %METHOD: GET LOG DENSITY
        %PARAMETER:
            %x: scalar, compound poisson variable
        %RETURN:
            %ln_pdf: log density
        function ln_pdf = getlnpdf(this,x)
            %declare array of terms to be summed
            terms = zeros(1,3);
            %add the exponential terms term
            terms(1) = -this.lambda-x*this.beta;
            %if x is bigger than 0, add the gamma terms
            if x>0
                terms(2) = -log(x);
                terms(3) = this.lnSumW(x,0);
            end
            %return the log density
            ln_pdf = sum(terms);
        end
        
        %METHOD: GET MARGINIAL LOG LIKELIHOOD
        %RETURN:
            %lnL: marginial log likelihood
        function lnL = getMarginallnL(this)
            %declare array of log likelihood terms, one for each data point
            lnL_terms = zeros(1,this.n);
            %for each data point
            for i = 1:this.n
                %get the data
                x = this.X(i);
                %get the log density and append it to lnL_terms
                lnL_terms(i) = this.getlnpdf(x);
            end
            %sum the log likelihood terms
            lnL = sum(lnL_terms);
        end
        
        %METHOD: GET FISHER'S INFORMATION MATRIX
        %Returns the Fisher's information matrix (using the joint log likelihood)
        function I = getFisherInformation(this)
            
            %declare 3 x 3 matrix
            I = zeros(3,3);
            
            %set for lambda
            I(1,1) = this.n/this.lambda;
            %set for alpha, the expectation was approximated using the delta method
            I(2,2) = this.n * sum([
                psi(1,this.lambda*this.alpha)*(this.lambda^2+this.lambda);
                2*psi(2,this.lambda*this.alpha)*this.alpha*this.lambda^2;
                0.5*psi(3,this.lambda*this.alpha)*this.alpha^2*this.lambda^3
            ]);
            %set for the covariance between alpha and beta
            I(2,3) = -this.n*this.lambda/this.beta;
            I(3,2) = I(2,3);
            %set for beta
            I(3,3) = this.n*this.lambda*this.alpha/(this.beta^2);
            
        end
        
        %METHOD: GET OBJECTIVE FUNCTION FOR THE M STEP
        %RETURN
            %T: conditional expectation of the joint log likelihood
        function T = getMObjective(this)
        %declare array of log likelihood terms, one for each data point
        T = zeros(7,this.n);
        %for each data point
        for i = 1:this.n
            %get the observable and latent variable
            x = this.X(i);
            y = this.Y(i);
            %if x is bigger than 0
            if x > 0
                %add conditional expectation terms
                T(1,i) = y*this.alpha*log(this.beta);
                T(2,i) = -gammaln(y*this.alpha);
                T(3,i) = y*this.alpha*log(x);
                T(4,i) = -x*this.beta;
                T(7,i) = -0.5*y*this.alpha^2*psi(1,y*this.alpha);
            end
            %add poisson terms
            T(5,i) = y*log(this.lambda);
            T(6,i) = -this.lambda;
        end
        %sum up all the terms
        T = sum(sum(T));
        end
        
        %METHOD: E STEP
        %Updates the member variables Y and Y_var given X, lambda, alpha and beta
        %Y and Y_var are updated using the conditional expectations
        function EStep(this)
            %for each data point
            for i = 1:this.n
                %get the observable
                x = this.X(i);
                %if the observable is 0, then y is 0
                if x == 0
                    y = 0;
                    var = nan;
                %else estimate the mean and variance
                else
                    %work out the normalisation constant for expectations
                    normalisation_constant = this.lnSumW(x, 0);
                    %work out the expectation
                    y = exp(this.lnSumW(x, 1) - normalisation_constant);
                    %work out the variance
                    var = exp(this.lnSumW(x, 2) - normalisation_constant) - y^2;
                end
                %assign the expectation and variance
                this.Y(i) = y;
                this.Y_var(i) = var;
            end
        end

        %METHOD: M STEP
        %Update the member variables lambda, alpha and beta given X, Y and Y_var
        function MStep(this)

            %update the poisson variable
            mean_Y = mean(this.Y);

            %get non zero variables
            X_0 = this.X(this.X~=0);
            Y_0 = this.Y(this.Y~=0);
            var_0 = this.Y_var(~isnan(this.Y_var));

            %work out the gradient
            d_alpha_lnL = sum(Y_0*log(this.beta) - Y_0.*psi(this.alpha*Y_0) + Y_0.*log(X_0) - (0.5*this.alpha^2).*var_0.*Y_0.*psi(2,Y_0*this.alpha) - this.alpha*var_0.*psi(1,Y_0*this.alpha));
            d_beta_lnL = sum(Y_0*this.alpha/this.beta - X_0);

            %work out the Hessian
            d_alpha_beta_lnL = sum(Y_0/this.beta);
            d_alpha_alpha_lnL = -sum( Y_0.^2.*psi(1,this.alpha*Y_0) + var_0.*psi(1,this.alpha*Y_0) + (2*this.alpha).*var_0.*Y_0.*psi(2,this.alpha*Y_0) + (0.5*this.alpha^2).*var_0.*Y_0.^2.*psi(3,Y_0*this.alpha) );
            d_beta_beta_lnL = -sum(Y_0*this.alpha/(this.beta^2));

            %put all the variables together in vector and matrix form
            theta = [this.alpha; this.beta]; %parameter matrix
            del_lnL = [d_alpha_lnL; d_beta_lnL]; %gradient
            H = [d_alpha_alpha_lnL, d_alpha_beta_lnL; d_alpha_beta_lnL, d_beta_beta_lnL]; %Hessian
            
            %do a newton step
            theta = theta - H\del_lnL;
            
%             %STIRLING'S APPROXIMATION
%             this.alpha = exp(sum([
%                 sum(Y_0.*(log(Y_0)-log(X_0)));
%                 sum(this.X)*(log(sum(X_0))-log(sum(Y_0)));
%                 ])/(sum(X_0)-sum(Y_0)));
%             this.beta = this.alpha*sum(Y_0)/sum(X_0);
%             theta = [this.alpha; this.beta]; %parameter matrix

            %if any of the parameters are negative, throw error
            if any(theta<0)
                error('negative parameter');
            end

            %update the parameters
            this.setParameters(mean_Y, theta(1), theta(2));

        end
        
        %METHOD: YMAX
        %Gets the index of the biggest term in the compound Poisson sum
        %PARAMETERS
            %x: scalar, compound poisson random variable
        %RETURN
            %y_max: positive integer, index of the biggest term in the compound Poisson sum
        function y_max = yMax(this, x)
            %get the optima with respect to the sum index, then round it to get an integer
            y_max = round(exp( (2-this.p)*log(x) - log(this.phi) - log(2-this.p) ));
            %if the integer is 0, then set the index to 1
            if y_max == 0
                y_max = 1;
            end
        end
        
        %METHOD: lnWy
        %Return a log term from the compound Poisson sum
        %PARAMETERS:
            %x: scalar, compound poisson variable
            %y: positive integer, term index of the compound Poisson sum
            %phi: dispersion parameter
            %p: power index
        %RETURN:
            %ln_Wy: log compopund Poisson term
        function ln_Wy = lnWy(this, x, y)
            terms = zeros(1,6); %declare array of terms to be summed to work out ln_Wy
            %work out each individual term
            terms(1) = -y*this.alpha*log(this.p-1);
            terms(2) = y*this.alpha*log(x);
            terms(3) = -y*(1+this.alpha)*log(this.phi);
            terms(4) = -y*log(2-this.p);
            terms(5) = -gammaln(1+y);
            terms(6) = -gammaln(this.alpha*y);
            %sum the terms to get the log compound Poisson sum term
            ln_Wy = sum(terms);
        end
        
        
        %METHOD: lnSumW (log compound sum)
        %Works out the compound Poisson sum, only important terms are summed
        %PARAMETERS
            %x: scalar, compound Poisson variable
            %y_power: 0,1,2,..., used for taking the sum for y^y_power * W_y which is used for taking expectations
            %lambda: poisson parameter
            %alpha: gamma shape parameter
            %beta: gamma rate parameter
        %RETURN:
            %ln_sum_w: log compound Poisson sum
        function [ln_sum_w, y_l, y_u] = lnSumW(this, x, y_pow)

            %get the y with the biggest term in the compound Poisson sum
            y_max = this.yMax(x);
            %get the biggest log compound Poisson term + any expectation terms
            ln_w_max = this.lnWy(x,y_max) + y_pow*log(y_max);

            %declare array of compound poisson terms
            %each term is a ratio of the compound poisson term with the maximum compound poisson term
            terms = zeros(1,this.n_compound_poisson_term);
            %the first term is 1 (w_max / w_max = 1);
            terms(1) = 1;

            %declare booleans got_y_l and got_y_u
            %these are true with we got the lower and upper bound respectively for the compound Poisson sum
            got_y_l = false;
            got_y_u = false;

            %declare the summation bounds, y_l for the lower bound, y_u for the upper bound
            y_l = y_max;
            y_u = y_max;
            
            %declare a counter which counts the number of compound Poisson terms stored in the array terms
            counter = 2;

            %calculate the compound poisson terms starting at y_l and working downwards
            %if the lower bound is 1, can't go any lower and set got_y_l to be true
            if y_l == 1
                got_y_l = true;
            end
            
            %while we haven't got a lower bound
            while ~got_y_l
                %lower the lower bound
                y_l = y_l - 1;
                %if the lower bound is 0, then set got_y_l to be true and raise the lower bound by one
                if y_l == 0
                    got_y_l = true;
                    y_l = y_l + 1;
                %else the lower bound is not 0
                else
                    %calculate the log ratio of the compound poisson term with the maximum compound poisson term
                    log_ratio = sum([this.lnWy(x,y_l), y_pow*log(y_l), -ln_w_max]);
                    %if this log ratio is bigger than the threshold
                    if log_ratio > this.compound_poisson_sum_threshold
                        %append the ratio to the array of terms
                        terms(counter) = exp(log_ratio);
                        %raise the counter by 1
                        counter = counter + 1;
                        %if the array has exceeded memory allocation, throw error
                        if counter > this.n_compound_poisson_term
                            error('Number of terms exceed memory allocation');
                        end
                    %else the log ratio is smaller than the threshold
                    %set got_y_l to be true and raise the lower bound by 1
                    else
                        got_y_l = true;
                        y_l = y_l + 1;
                    end       
                end 
            end

            %while we haven't got an upper bound
            while ~got_y_u
                %raise the upper bound by 1
                y_u = y_u + 1;
                %calculate the log ratio of the compound poisson term with the maximum compound poisson term
                log_ratio = sum([this.lnWy(x,y_u), y_pow*log(y_u), -ln_w_max]);
                %if this log ratio is bigger than the threshold
                if log_ratio > this.compound_poisson_sum_threshold
                    %append the ratio to the array of terms
                    terms(counter) = exp(log_ratio);
                    %raise the counter by 1
                    counter = counter + 1;
                    %if the array has exceeded memory allocation, throw error
                    if counter > this.n_compound_poisson_term
                        error('Number of terms exceed memory allocation');
                    end
                %else the log ratio is smaller than the threshold
                %set got_y_u to be true and lower the upper bound by 1
                else
                    got_y_u = true;
                    y_u = y_u - 1;
                end       
            end
            
            %lower the counter by 1
            counter = counter - 1;
            %work out the compound Poisson sum using non-zero terms
            ln_sum_w = ln_w_max + log(sum(terms(1:counter)));

        end
        
        %METHOD: GET INVERSE CDF
        %Using the trapezium rule to get the cdf then interpolate it to get the inverse
        %PARAMETERS:
            %p_array: order array of parameters of the inverse cdf
            %a: start limit value of numerical integration
            %b: end limit value of numerical integration
            %n: number of trapeziums
            %use_saddle: boolean, false to use exact pdf, else use saddle point approximated pdf
        %NOTES:
            %the saddle point approximated pdf does not support exactly 0
        %RETURN:
            %x: array of compound poisson variables, one for each element in p_array
                %corresponding to the inverse cdf
        function x = getInvCdf(this,p_array, a, b, n)

            %get x coordinates for each trapezium
            x_array = (linspace(a,b,n));
            %get the hight of each trapezium
            h = (b-a)/n;
            
            %for each x, get the pdf evaluation
            pdf_array = this.getPdf(x_array);
            
            %declare array of cdf, one for each element in x_array
            cdf_array = zeros(n,1);
            
            %get the first cdf evaluation
            %if the lower limit is 0, the cdf = pdf because there is probability mass at 0
            if a==0
                cdf_array(1) = pdf_array(1);
            %else start the cdf at a
            else
                cdf_array(1) = 0;
            end
            
            %then for each trapezium, do numerical integration
            for i = 2:n
                %add the area of the trapzium and append it to cdf_array
                cdf_array(i) = cdf_array(i-1) + 0.5*h*(pdf_array(i)+pdf_array(i-1));
            end
            
            %declare array of compound poisson variables
            %one for each element in p_array
            x = zeros(numel(p_array),1);
            
            %declare a variable for counting the number of inverse cdf calculated
            %this corresponds to the pointer for the array x
            counter = 1;
            
            %for each trapezium
            for i = 1:n
                
                %while the current probability in p_array is less than the cdf of the current trapezium
                %then the inverse cdf can be calculated using interpolation
                while p_array(counter) < cdf_array(i)
                    
                    %if this is the first trapezium
                    %then the inverse cdf is the lower limit of the integration
                    if i == 1
                        x(counter) = x_array(i);
                    %else this is not the first trapezium
                    %work out the inverse cdf using interpolation
                    else
                        x(counter) = x_array(i-1) + (x_array(i)-x_array(i-1))*(p_array(counter)-cdf_array(i-1))/(cdf_array(i)-cdf_array(i-1));
                    end
                    
                    %increase the p_array counter by one
                    counter = counter + 1;
                    %if the counter exceed all elements in p_array, break the while loop
                    if counter > numel(p_array)
                        return;
                    end %if counter
                    
                end %while
                
            end %for i

            %if not all inverse cdf have been calculated, they exceed the upper limit of the integration
            %return infinite compound poisson variables
            if counter <= numel(p_array)
                x(counter:end) = inf;
            end
            
        end %getinv
        
    end %methods

    %STATIC METHODS
    methods (Static)

        %STATIC METHOD: SIMULATE
        %Simulate the compound Poisson random variable
        %PARAMETERS:
            %lambda: Poisson parameter
            %alpha: Gamma shape parameter
            %neta: Gamma rate parameter
        %RETURN:
            %X: vector of observables
            %Y: vector of latent variables
        function [X,Y] = simulate(n,lambda,alpha,beta)
            Y = poissrnd(lambda,n,1); %simulate latent poisson variables
            X = gamrnd(Y*alpha,1/beta); %simulate observable gamma
        end

    end

end