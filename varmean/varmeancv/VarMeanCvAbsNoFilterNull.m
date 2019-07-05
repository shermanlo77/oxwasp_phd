classdef VarMeanCvAbsNoFilterNull < VarMeanCv
  
  methods (Access = protected)
    
    function setup(this)
      this.setup@VarMeanCv(AbsNoFilterDeg120(), uint32(3293056508));
    end
    
  end
  
end
