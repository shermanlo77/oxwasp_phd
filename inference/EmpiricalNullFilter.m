%CLASS: EMPIRICAL NULL FILTER
%Does the empirical null filter, makes use of ImageJ and multiple threads
%HOW TO USE:
  %Instantiate this by passing a radius
  %Set advanced options if desired using methods for example setNStep
  %Call the method filter and pass the image you want to filter
  %Call getter methods getFilteredImage, getNullMean and/or getNullStd to get the results
classdef EmpiricalNullFilter < handle
  
  properties (SetAccess = private)
    radius; %radius of the kernel
    filteredImage; %resulting filtered image
    nullMean; %empirical null mean image
    nullStd; %empirical null std image
    javaFilter; %the java object EmpiricalNullFilter
  end
  
  methods (Access = public)
    
    %CONSTRUCTOR
    %PARAMETERS:
      %radius: the radius of the kernel
    function this = EmpiricalNullFilter(radius)
      this.radius = radius;
      this.javaFilter = uk.ac.warwick.sip.empiricalnullfilter.EmpiricalNullFilter();
      this.javaFilter.setRadius(radius);
    end
    
    %METHOD: FILTER
    %Empirical null filter on this image
    %Also produces the empirical null mean and empirical null std image
    %These images are obtained using the getter methods
    function filter(this, image)
      this.javaFilter.setNPasses(1);
      [height, width] = size(image);
      this.javaFilter.filter(single(image));
      this.filteredImage = reshape(this.javaFilter.getFilteredImage(), height, width);
      this.nullMean = reshape(this.javaFilter.getOutputImage( ...
          uk.ac.warwick.sip.empiricalnullfilter.EmpiricalNullFilter.NULL_MEAN), height, width);
      this.nullStd = reshape(this.javaFilter.getOutputImage( ...
          uk.ac.warwick.sip.empiricalnullfilter.EmpiricalNullFilter.NULL_STD), height, width);      
    end
    
    %METHOD: GET FILTERED IMAGE
    function filteredImage = getFilteredImage(this)
      filteredImage = this.filteredImage;
    end
    
    %METHOD: GET NULL MEAN IMAGE
    function nullMean = getNullMean(this)
      nullMean = this.nullMean;
    end
    
    %METHOD: GET NULL STD IMAGE
    function nullStd = getNullStd(this)
      nullStd = this.nullStd;
    end
    
    %METHOD: SET N INITIAL
    %Set the number of times to do newton-raphson
    function setNInitial(this, nInitial)
      this.javaFilter.setNInitial(nInitial);
    end
    
    %METHOD: GET N INITIAL
    %Get the number of times to do newton-raphson
    function nInitial = getNInital(this)
      nInitial = this.javaFilter.getNInitial();
    end
    
    %METHOD: SET N STEP
    %Set the number of steps to do newton-raphson
    function setNStep(this, nStep)
      this.javaFilter.setNStep(nStep);
    end
    
    %METHOD: GET N STEP
    %Get the number of steps to do newton-raphson
    function nStep = getNStep(this)
      nStep = this.javaFilter.getNStep();
    end
    
    %METHOD: Set LOG 10 TOLERANCE
    %Set the tolerance used in the stopping condition in newton-raphson
    %The stopping condition is (Math.abs(dxLnF[1])<tolerance) where dxLnF is the first diff of the
      %log density
    function setLog10Tolerance(this, log10Tolerance)
      this.javaFilter.setLog10Tolerance(log10Tolerance);
    end
    
    %METHOD: GET LOG 10 TOLERANCE
    %Get the tolerance used in the stopping condition in newton-raphson
    %The stopping condition is (Math.abs(dxLnF[1])<tolerance) where dxLnF is the first diff of the
      %log density
    function log10Tolerance = getLog10Tolerance(this)
      log10Tolerance = this.javaFilter.getLog10Tolerance();
    end
    
    %METHOD: SET BANDWIDTH A
    %Set the parameter A where the bandwidth used for the density estimate is
      %bandwidthParameterB * min(dataStd, iqr/1.34)* (n^-0.2) + bandwidthParameterA
    function setBandwidthA(this, bandwidthParameterA)
      this.javaFilter.setBandwidthA(bandwidthParameterA);
    end
    
    %METHOD: GET BANDWIDTH A
    %Get the parameter A where the bandwidth used for the density estimate is
      %bandwidthParameterB * min(dataStd, iqr/1.34)* (n^-0.2) + bandwidthParameterA
    function bandwidthParameterA = getBandwidthA(this)
      bandwidthParameterA = this.javaFilter.getBandwidthA();
    end
    
    %METHOD: SET BANDWIDTH B
    %Set the parameter B where the bandwidth used for the density estimate is
      %bandwidthParameterB * min(dataStd, iqr/1.34)* (n^-0.2) + bandwidthParameterA
    function setBandwidthB(this, bandwidthParameterB)
      this.javaFilter.setBandwidthA(bandwidthParameterB);
    end
    
    %METHOD: GET BANDWIDTH B
    %Get the parameter B where the bandwidth used for the density estimate is
      %bandwidthParameterB * min(dataStd, iqr/1.34)* (n^-0.2) + bandwidthParameterA
    function bandwidthParameterB = getBandwidthB(this)
      bandwidthParameterB = this.javaFilter.getBandwidthB();
    end
    
    %METHOD: SET PROGRESS BAR
    %Turn the progress bar on or off, by default it is off
    %PARAMETERS:
      %showProgressBar: boolean, true for the progress bar to be on
    function setProgress(this, showProgressBar)
      this.javaFilter.setProgress(showProgressBar);
    end
  
  end
  
end

