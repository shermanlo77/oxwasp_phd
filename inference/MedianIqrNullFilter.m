%MIT License
%Copyright (c) 2019 Sherman Lo

%CLASS: MEDIAN IQR NULL FILTER
%See superclass EmpiricalNullFilter
%Does the empirical null filter, makes use of ImageJ and multiple threads
%Replaces empirical null mean with median
%Replaces empirical null std with iqr/1.349
classdef MedianIqrNullFilter < EmpiricalNullFilter
  
  methods (Access = public)
    
    %CONSTRUCTOR
    %PARAMETERS:
      %radius: the radius of the kernel
    function this = MedianIqrNullFilter(radius)
      this@EmpiricalNullFilter(radius);
      this.javaFilter = uk.ac.warwick.sip.empiricalnullfilter.MedianIqrNullFilter();
      this.javaFilter.setRadius(radius);
    end
    
  end
  
end

