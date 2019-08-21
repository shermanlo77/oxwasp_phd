%MIT License
%Copyright (c) 2019 Sherman Lo

%CLASS: MAD MODE NULL FILTER
%See superclass EmpiricalNullFilter
%Does the empirical null filter, makes use of ImageJ and multiple threads
%Replaces empirical null std with median around the mode x 1.4826
classdef MadModeNullFilter < EmpiricalNullFilter
  
  methods (Access = public)
    
    function this = MadModeNullFilter(radius)
      this@EmpiricalNullFilter(radius);
      this.javaFilter = uk.ac.warwick.sip.empiricalnullfilter.MadModeNullFilter();
      this.javaFilter.setRadius(radius);
    end
    
  end
  
end

