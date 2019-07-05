classdef GlmSelectAicAbsFilterBw < GlmSelectAic
  
  properties
  end
  
  methods (Access = protected)
    
    function setup(this)
      this.setup@GlmSelectAic(AbsFilterDeg120(), uint32(1757771692));
      this.scan.addShadingCorrectorBw();
    end
    
  end
  
end