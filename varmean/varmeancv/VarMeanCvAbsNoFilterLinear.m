classdef VarMeanCvAbsNoFilterLinear < VarMeanCv
  
  methods (Access = protected)
    
    function setup(this)
      this.setup@VarMeanCv(AbsNoFilterDeg120(), uint32(889800029));
      this.scan.addShadingCorrectorLinear();
    end
    
  end
  
end
