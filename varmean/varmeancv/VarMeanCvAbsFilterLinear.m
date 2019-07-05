classdef VarMeanCvAbsFilterLinear < VarMeanCv
  
  methods (Access = protected)
    
    function setup(this)
      this.setup@VarMeanCv(AbsFilterDeg120(), uint32(2108618975));
      this.scan.addShadingCorrectorLinear();
    end
    
  end
  
end
