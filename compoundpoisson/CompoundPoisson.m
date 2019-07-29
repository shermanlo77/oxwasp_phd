classdef CompoundPoisson < handle
  
  %MEMBER VARIABLES
  properties (SetAccess = protected)
    lambda; %poisson parameter
    alpha; %gamma shape parameter
    beta; %gamma rate parameter
    phi; %dispersion parameter
    p; %power index
    mu; %mean
    sigma; %standard deviation
    X; %array of observed compound poisson variables
    Y; %array of latent poisson variables
    YVar; %array of latent poisson variances
    n; %number of data points
    %maximum number of terms to be calculated in the compound poisson sum
    nCompoundPoissonTerm = 1E7;
    
    compoundPoissonSumThreshold = -37; %negative number
    %if ln(compound poisson term / biggest compound poisson term) > compoundPoissonSumThreshold
    %then that term is considered for the compound poisson sum
    
    isSupportZeroMass = true; %boolean, ture if can support probability mass at 0
  end
  
  %METHODS
  methods
    
    %CONSTRUCTOR
    function this = CompoundPoisson()
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
      this.YVar = zeros(numel(this.X),1);
    end
    
    %METHOD: SET PARAMETERS
    %Assign the member variables lambda, alpha and beta
    %PARMAETERS:
      %lambda: poisson parameter
      %alpha: gamma shape parameter
      %beta: gamma rate parameter
    function setParameters(this, lambda, alpha, beta)
      this.lambda = lambda;
      this.alpha = alpha;
      this.beta = beta;
      this.p = (alpha+2)/(alpha+1);
      this.phi = (1+alpha)*beta^(this.p-2)*(alpha*lambda)^(1-this.p);
      this.mu = this.lambda * this.alpha / this.beta;
      this.sigma = sqrt(this.phi * this.mu^this.p);
    end
    
    %METHODS: SET N
    %Set the number of data, this is required to evaluate the Fisher's information matrix
    function setN(this, n)
      this.n = n;
    end
    
    %METHOD: GET DENSITY
    %PARAMETER:
      %x: vector of compound Poisson variables
    %RETURN:
      %pdf: vector of densities, for each element in x
    function pdf = getPdf(this,x)
      pdf = zeros(numel(x),1);
      for i = 1:numel(x)
        pdf(i) = exp(this.getLnPdf(x(i)));
      end
    end
    
    %METHOD: GET LOG DENSITY
    %PARAMETER:
      %x: scalar, compound poisson variable
    %RETURN:
      %lnPdf: log density
    function lnPdf = getLnPdf(this,x)
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
      lnPdf = sum(terms);
    end
    
    %METHOD: GET MARGINIAL LOG LIKELIHOOD
    %RETURN:
      %lnL: marginial log likelihood
    function lnL = getMarginallnL(this)
      %declare array of log likelihood terms, one for each data point
      lnLTerms = zeros(1,this.n);
      %for each data point
      for i = 1:this.n
        %get the data
        x = this.X(i);
        %get the log density and append it to lnLTerms
        lnLTerms(i) = this.getLnPdf(x);
      end
      %sum the log likelihood terms
      lnL = sum(lnLTerms);
    end
    
    %METHOD: GET FISHER'S INFORMATION MATRIX
    %Returns the Fisher's information matrix (using the joint log likelihood)
    %NOTE: this.n needs to be set, call the method setN to do this
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
    %Updates the member variables Y and YVar given X, lambda, alpha and beta
    %Y and YVar are updated using the conditional expectations
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
          normalisationConstant = this.lnSumW(x, 0);
          %work out the expectation
          y = exp(this.lnSumW(x, 1) - normalisationConstant);
          %work out the variance
          var = exp(this.lnSumW(x, 2) - normalisationConstant) - y^2;
        end
        %assign the expectation and variance
        this.Y(i) = y;
        this.YVar(i) = var;
      end
    end
    
    %METHOD: M STEP
    %Update the member variables lambda, alpha and beta given X, Y and YVar
    function MStep(this)
      
      %update the poisson variable
      YMean = mean(this.Y);
      
      %get non zero variables
      X0 = this.X(this.X~=0);
      Y0 = this.Y(this.Y~=0);
      var0 = this.YVar(~isnan(this.YVar));
      
      %work out the gradient
      dAlphaLnL = sum( ...
          Y0*log(this.beta)...
          -Y0.*psi(this.alpha*Y0)...
          + Y0.*log(X0)...
          - (0.5*this.alpha^2).*var0.*Y0.*psi(2,Y0*this.alpha)...
          - this.alpha*var0.*psi(1,Y0*this.alpha));
      dBetaLnL = sum(Y0*this.alpha/this.beta - X0);
      
      %work out the Hessian
      dAlphaBetaLnL = sum(Y0/this.beta);
      dAlphaAlphaLnL = -sum(...
          Y0.^2.*psi(1,this.alpha*Y0)...
          + var0.*psi(1,this.alpha*Y0)...
          + (2*this.alpha).*var0.*Y0.*psi(2,this.alpha*Y0)...
          + (0.5*this.alpha^2).*var0.*Y0.^2.*psi(3,Y0*this.alpha));
      dBetaBetaLnL = -sum(Y0*this.alpha/(this.beta^2));
      
      %put all the variables together in vector and matrix form
      theta = [this.alpha; this.beta]; %parameter matrix
      delLnL = [dAlphaLnL; dBetaLnL]; %gradient
      H = [dAlphaAlphaLnL, dAlphaBetaLnL; dAlphaBetaLnL, dBetaBetaLnL]; %Hessian
      
      %do a newton step
      theta = theta - H\delLnL;
      
      %if any of the parameters are negative, throw error
      if any(theta<0)
        error('negative parameter');
      end
      
      %update the parameters
      this.setParameters(YMean, theta(1), theta(2));
      
    end
    
    %METHOD: YMAX
    %Gets the index of the biggest term in the compound Poisson sum
    %PARAMETERS
      %x: scalar, compound poisson random variable
    %RETURN
      %yMax: positive integer, index of the biggest term in the compound Poisson sum
    function yMax = yMax(this, x)
      %get the optima with respect to the sum index, then round it to get an integer
      yMax = round(exp( (2-this.p)*log(x) - log(this.phi) - log(2-this.p) ));
      %if the integer is 0, then set the index to 1
      if yMax == 0
        yMax = 1;
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
      %lnWy: log compopund Poisson term
    function lnWy = lnWy(this, x, y)
      terms = zeros(1,6); %declare array of terms to be summed to work out lnWy
      %work out each individual term
      terms(1) = -y*this.alpha*log(this.p-1);
      terms(2) = y*this.alpha*log(x);
      terms(3) = -y*(1+this.alpha)*log(this.phi);
      terms(4) = -y*log(2-this.p);
      terms(5) = -gammaln(1+y);
      terms(6) = -gammaln(this.alpha*y);
      %sum the terms to get the log compound Poisson sum term
      lnWy = sum(terms);
    end
    
    %METHOD: lnSumW (log compound sum)
    %Works out the compound Poisson sum, only important terms are summed
    %PARAMETERS
      %x: scalar, compound Poisson variable
      %yPow: 0,1,2,..., used for taking the sum for y^yPow * Wy which is used for taking
          %expectations
      %lambda: poisson parameter
      %alpha: gamma shape parameter
      %beta: gamma rate parameter
    %RETURN:
      %lnSumW: log compound Poisson sum
      %yL: upper sum element
      %yU: lower sum element
    function [lnSumW, yL, yU] = lnSumW(this, x, yPow)
      
      %get the y with the biggest term in the compound Poisson sum
      yMax = this.yMax(x);
      %get the biggest log compound Poisson term + any expectation terms
      lnWMax = this.lnWy(x,yMax) + yPow*log(yMax);
      
      %declare array of compound poisson terms
      %each term is a ratio of the compound poisson term with the maximum compound poisson term
      terms = zeros(1,this.nCompoundPoissonTerm);
      %the first term is 1
      terms(1) = 1;
      
      %declare booleans isGotYL and isGotYU
      %these are true with we got the lower and upper bound respectively for the compound Poisson
          %sum
      isGotYL = false;
      isGotYU = false;
      
      %declare the summation bounds, yL for the lower bound, yU for the upper bound
      yL = yMax;
      yU = yMax;
      
      %declare a counter which counts the number of compound Poisson terms stored in the array terms
      counter = 2;
      
      %calculate the compound poisson terms starting at yL and working downwards
      %if the lower bound is 1, can't go any lower and set isGotYL to be true
      if yL == 1
        isGotYL = true;
      end
      
      %while we haven't got a lower bound
      while ~isGotYL
        %lower the lower bound
        yL = yL - 1;
        %if the lower bound is 0, then set isGotYL to be true and raise the lower bound by one
        if yL == 0
          isGotYL = true;
          yL = yL + 1;
          %else the lower bound is not 0
        else
          %calculate the log ratio of the compound poisson term with the maximum compound poisson
              %term
          logRatio = sum([this.lnWy(x,yL), yPow*log(yL), -lnWMax]);
          %if this log ratio is bigger than the threshold
          if logRatio > this.compoundPoissonSumThreshold
            %append the ratio to the array of terms
            terms(counter) = exp(logRatio);
            %raise the counter by 1
            counter = counter + 1;
            %if the array has exceeded memory allocation, throw error
            if counter > this.nCompoundPoissonTerm
              error('Number of terms exceed memory allocation');
            end
            %else the log ratio is smaller than the threshold
            %set isGotYL to be true and raise the lower bound by 1
          else
            isGotYL = true;
            yL = yL + 1;
          end
        end
      end
      
      %while we haven't got an upper bound
      while ~isGotYU
        %raise the upper bound by 1
        yU = yU + 1;
        %calculate the log ratio of the compound poisson term with the maximum compound poisson term
        logRatio = sum([this.lnWy(x,yU), yPow*log(yU), -lnWMax]);
        %if this log ratio is bigger than the threshold
        if logRatio > this.compoundPoissonSumThreshold
          %append the ratio to the array of terms
          terms(counter) = exp(logRatio);
          %raise the counter by 1
          counter = counter + 1;
          %if the array has exceeded memory allocation, throw error
          if counter > this.nCompoundPoissonTerm
            error('Number of terms exceed memory allocation');
          end
          %else the log ratio is smaller than the threshold
          %set isGotYU to be true and lower the upper bound by 1
        else
          isGotYU = true;
          yU = yU - 1;
        end
      end
      
      %lower the counter by 1
      counter = counter - 1;
      %work out the compound Poisson sum using non-zero terms
      lnSumW = lnWMax + log(sum(terms(1:counter)));
      
    end
    
    %METHOD: GET INVERSE CDF
    %Using the trapezium rule to get the cdf then interpolate it to get the inverse
    %PARAMETERS:
      %pArray: order array of parameters of the inverse cdf
      %a: start limit value of numerical integration
      %b: end limit value of numerical integration
      %n: number of trapeziums
    %NOTES:
      %the saddle point approximated pdf does not support exactly 0
    %RETURN:
      %x: array of compound poisson variables, one for each element in pArray corresponding to the
          %inverse cdf
    function x = getInvCdf(this, pArray, a, b, n)
      
      %get x coordinates for each trapezium
      xArray = (linspace(a,b,n));
      %get the hight of each trapezium
      h = (b-a)/n;
      
      %for each x, get the pdf evaluation
      pdfArray = this.getPdf(xArray);
      
      %declare array of cdf, one for each element in xArray
      cdfArray = zeros(n,1);
      
      %get the first cdf evaluation
      %if the lower limit is 0, the cdf = pdf because there is probability mass at 0
      if a==0
        cdfArray(1) = pdfArray(1);
        %else start the cdf at a
      else
        cdfArray(1) = 0;
      end
      
      %then for each trapezium, do numerical integration
      for i = 2:n
        %add the area of the trapzium and append it to cdfArray
        cdfArray(i) = cdfArray(i-1) + 0.5*h*(pdfArray(i)+pdfArray(i-1));
      end
      
      %declare array of compound poisson variables
      %one for each element in pArray
      x = zeros(numel(pArray),1);
      
      %declare a variable for counting the number of inverse cdf calculated
      %this corresponds to the pointer for the array x
      counter = 1;
      
      %for each trapezium
      for i = 1:n
        
        %while the current probability in pArray is less than the cdf of the current trapezium
        %then the inverse cdf can be calculated using interpolation
        while pArray(counter) < cdfArray(i)
          
          %if this is the first trapezium
          %then the inverse cdf is the lower limit of the integration
          if i == 1
            x(counter) = xArray(i);
            %else this is not the first trapezium
            %work out the inverse cdf using interpolation
          else
            x(counter) = xArray(i-1) + ...
                (xArray(i)-xArray(i-1))*(pArray(counter)-cdfArray(i-1))...
                /(cdfArray(i)-cdfArray(i-1));
          end
          
          %increase the pArray counter by one
          counter = counter + 1;
          %if the counter exceed all elements in pArray, break the while loop
          if counter > numel(pArray)
            return;
          end %if counter
          
        end %while
        
      end %for i
      
      %if not all inverse cdf have been calculated, they exceed the upper limit of the integration
      %return infinite compound poisson variables
      if counter <= numel(pArray)
        x(counter:end) = inf;
      end
      
    end %getinv
    
    %METHOD: TO STRING
    function string = toString(this)
      string = strcat(class(this),'lambda',num2str(this.lambda), ...
          'alpha',num2str(this.alpha),'beta',num2str(this.beta));
    end
    
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