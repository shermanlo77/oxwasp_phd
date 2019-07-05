classdef VarMeanCvAbsFilterBw < VarMeanCv
  
  methods (Access = protected)
    
    function setup(this)
      this.setup@VarMeanCv(AbsFilterDeg120(), uint32(3157061918));
      this.scan.addShadingCorrectorBw();
    end
    
  end
  
end
