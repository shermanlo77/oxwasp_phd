%MIT License
%Copyright (c) 2019 Sherman Lo

%CLASS: MAD MODE NULL
%Wrapper class for the java class MadModeNull
%
%Same as EmpiricalNull but uses median around the mode for the null std
classdef MadModeNull < EmpiricalNull
  
  properties
  end
  
  methods (Access = public)
    
    function this = MadModeNull(zArray, initialValue, seed)
      this@EmpiricalNull(zArray, initialValue, seed);
    end
    
  end
  
  methods (Access = protected)
    
    function saveJavaObject(this, zArray, initialValue, seed)
      this.javaObj = uk.ac.warwick.sip.empiricalnullfilter.MadModeNull(zArray, ...
          initialValue, quantile(zArray,[0.25,0.5,0.75]), nanstd(zArray), sum(~isnan(zArray)), ...
          seed);
    end
    
  end
  
end

