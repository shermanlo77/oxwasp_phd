%MIT License
%Copyright (c) 2019 Sherman Lo

%COMPOUND POISSON NORMAL APPROXIMATION
%SEE SUPER CLASS COMPOUND POISSON
%pdf functions overriden to use normal approximation
classdef CompoundPoissonNorm < CompoundPoisson
  
  properties
  end
  
  methods
    
    %CONSTRUCTOR
    function this = CompoundPoissonNorm()
      %call superclass
      this@CompoundPoisson();
      this.isSupportZeroMass = false;
    end
    
    %OVERRIDE: GET DENSITY
    %PARAMETER:
    %x: vector of compound Poisson variables
    %RETURN:
    %pdf: vector of densities, for each element in x
    function pdf = getPdf(this,x)
      pdf = normpdf(x,this.mu,this.sigma);
    end
    
    %OVERRIDE: GET LOG DENSITY
    %PARAMETER:
    %x: scalar, compound poisson variable
    %RETURN:
    %ln_pdf: log density
    function ln_pdf = getlnpdf(this,x)
      ln_pdf = log(this.getPdf(x));
    end
    
    %OVERRIDE: GET INVERSE CDF
    function x = getInvCdf(this,pArray, ~,~,~)
      x = norminv(pArray,this.mu,this.sigma);
    end
    
  end
  
end

