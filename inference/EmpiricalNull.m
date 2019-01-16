%CLASS: EMPIRICAL NULL
%Wrapper class for the java class EmpiricalNull
%Does the empirical null analysis on an array of data
%HOW TO USE:
  %Pass the data to the constructor, set initial value for Newton Raphson
  %Call the method estimateNull()
  %Call the method getNullMean() and getNullStd() to get the empirical null parameters
%Various setter methods to set the options for the empirical null
classdef EmpiricalNull < handle
  
  properties (GetAccess = private)
    javaEmpiricalNull; %uk.ac.warwick.sip.empiricalnullfilter.EmpiricalNull object
  end
  
  methods (Access = public)
    
    %CONSTRUCTOR
    %PARAMETERS:
      %zArray: 1d array of z statistics
      %initialValue: value to start the newton raphson from
      %seed: long for setting the random number generator, used for using different initial values
    function this = EmpiricalNull(zArray, initialValue, seed)
      this.javaEmpiricalNull = uk.ac.warwick.sip.empiricalnullfilter.EmpiricalNull(zArray, ...
          initialValue, quantile(zArray,[0.25,0.5,0.75]), nanstd(zArray), sum(~isnan(zArray)), ...
          seed);
    end
    
    %METHOD: ESTIMATE NULL
    %Get the empirical null parameters
    function estimateNull(this)
      this.javaEmpiricalNull.estimateNull();
    end
    
    %METHOD: GET NULL MEAN
    function nullMean = getNullMean(this)
      nullMean = this.javaEmpiricalNull.getNullMean();
    end
    
    %METHOD: GET NULL STD
    function nullStd = getNullStd(this)
      nullStd = this.javaEmpiricalNull.getNullStd();
    end
    
    %=====GETTER AND SETTER METHODS HERE=====%
     % @param nInitial number of times to repeat the newton-raphson using different initial values
     % @param nStep number of steps in newton-raphson
     % @param log10Tolerance stopping condition tolerance for newton-raphson, stopping condition is
     %     Math.abs(dxLnF[1])<this.tolerance where dxLnF[1] is the gradient of the log density and
     %     this.tolerance is 10^log10Tolerance
     % @param bandwidthParameterA the bandwidth for the density estimate is
     %     B x 0.9 x std x n^{-1/5} + A
     % @param bandwidthParameterB the bandwidth for the density estimate is
     %     B x 0.9 x std x n^{-1/5} + A
    
    %METHOD: GET N INITIAL
    function nInitial = getNInitial(this)
      nInitial = this.javaEmpiricalNull.getNInitial();
    end
    
    %METHOD: SET N INITIAL
    function setNInitial(this, nInitial)
      this.javaEmpiricalNull.setNInitial(nInitial);
    end
    
    %METHOD: GET N STEP
    function nStep = getNStep(this)
      nStep = this.javaEmpiricalNull.getNStep();
    end
    
    %METHOD: SET N STEP
    function setNStep(this, nStep)
      this.javaEmpiricalNull.setNStep(nStep);
    end
    
    %METHOD: GET LOG 10 TOLERANCE
    function log10tolerance = getLog10Tolerance(this)
      log10tolerance = this.javaEmpiricalNull.getLog10Tolerance();
    end
    
    %METHOD: SET LOG 10 TOLERANCE
    function setLog10Tolerance(this, log10Tolerance)
      this.javaEmpiricalNull.setLog10Tolerance(log10Tolerance);
    end
    
    %METHOD: GET BANDWIDTH PARAMETER A
    function bandwidthParameterA = getBandwidthParameterA(this)
      bandwidthParameterA = this.javaEmpiricalNull.getBandwidthParameterA();
    end
    
    %METHOD: SET BANDWIDTH PARAMETER A
    function setBandwidthParameterA(this, bandwidthParameterA)
      this.javaEmpiricalNull.setBandwidthParameterA(bandwidthParameterA);
    end
    
    %METHOD: GET BANDWIDTH PARAMETER B
    function bandwidthParameterB = getBandwidthParameterB(this)
      bandwidthParameterB = this.javaEmpiricalNull.getBandwidthParameterB();
    end
    
    %METHOD: SET BANDWIDTH PARAMETER B
    function setBandwidthParameterB(this, bandwidthParameterB)
      this.javaEmpiricalNull.setBandwidthParameterB(bandwidthParameterB);
    end

  end
  
end
