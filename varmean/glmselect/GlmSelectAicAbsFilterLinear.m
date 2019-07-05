classdef GlmSelectAicAbsFilterLinear < GlmSelectAic
  
  properties
  end
  
  methods (Access = protected)
    
    function setup(this)
      this.setup@GlmSelectAic(AbsFilterDeg120(), uint32(657009667));
      this.scan.addShadingCorrectorLinear();
    end
    
  end
  
end