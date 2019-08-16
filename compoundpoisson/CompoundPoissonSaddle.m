%MIT License
%Copyright (c) 2019 Sherman Lo

%COMPOUND POISSON SADDLEPOINT APPROXIMATION
%SEE SUPER CLASS COMPOUND POISSON
  %overrides pdf methods to use saddlepoint approximation
classdef CompoundPoissonSaddle < CompoundPoisson
  
  properties
  end
  
  methods
    
    %CONSTRUCTOR
    function this = CompoundPoissonSaddle()
      %call superclass
      this@CompoundPoisson();
      this.isSupportZeroMass = false;
    end
    
    %OVERRIDE: GET DENSITY
    %For a range of the domain and given parameters, return the saddlepoint density. The
        %normalisation constant was worked out using the trapezium rule, with n_point - 1 strips
    %PARAMETERS:
      %x: row vector, equally spaced out compound poisson random variables
    %RETURN:
      %pdf: density evaluated at x
    function pdf = getPdf(this,x)
      %work out the log terms
      logTerms = sum([-(this.alpha+2)/(2*(this.alpha+1))*log(x);
        -x*this.beta;
        (x*this.beta/this.alpha)...
            .^(this.alpha/(this.alpha+1))*(this.lambda)^(1/(this.alpha+1))*(this.alpha+1)
        ]);
      
      %k is some constant to control over and under flow
      k = max(logTerms);
      %k = this.lambda;
      
      %work out the saddle density
      pdf = exp(logTerms - k);
      
      %work out the height of the trapziums
      h = (max(x)-min(x))/(numel(x)-1);
      %integrate the function
      area = 0.5*h*(pdf(1)+pdf(end)+2*sum(pdf(2:(end-1))));
      
      %normalise the saddle density
      pdf = pdf/area;
    end
    
    %OVERRIDE: GET LOG DENSITY
    %PARAMETER:
      %x: scalar, compound poisson variable
    %RETURN:
      %ln_pdf: log density
    function lnPdf = getlnpdf(this,x)
      lnPdf = log(this.getPdf(x));
    end
    
  end
  
end

