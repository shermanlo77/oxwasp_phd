classdef GlmSelectAicAbsNoFilterLinear < GlmSelectAic
  
  properties
  end
  
  methods (Access = protected)
    
    function setup(this)
      this.setup@GlmSelectAic(AbsNoFilterDeg120(), uint32(2133083960));
      this.scan.addShadingCorrectorLinear();
    end
    
  end
  
end