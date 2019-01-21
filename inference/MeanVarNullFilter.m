%CLASS: MEAN VARIANCE NULL FILTER
%See superclass EmpiricalNullFilter
%Does the empirical null filter, makes use of ImageJ and multiple threads
%Replaces empirical null mean with mean
%Replaces empirical null var with variance
classdef MeanVarNullFilter < EmpiricalNullFilter
  
  methods (Access = public)
    
    %CONSTRUCTOR
    %PARAMETERS:
      %radius: the radius of the kernel
    function this = MeanVarNullFilter(radius)
      this@EmpiricalNullFilter(radius);
      this.javaFilter = uk.ac.warwick.sip.empiricalnullfilter.MeanVarNullFilter();
      this.javaFilter.setRadius(radius);
    end
    
  end
  
end

