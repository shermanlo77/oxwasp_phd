classdef VarMeanCvAbsNoFilterBw < VarMeanCv
  
  methods (Access = protected)
    
    function setup(this)
      this.setup@VarMeanCv(AbsNoFilterDeg120(), uint32(1357762610));
      this.scan.addShadingCorrectorBw();
    end
    
  end
  
end
