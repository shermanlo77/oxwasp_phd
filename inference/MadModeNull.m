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

