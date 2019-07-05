classdef GlmSelectBicAbsFilterBw < GlmSelectBic
  
  properties
  end
  
  methods (Access = protected)
    
    function setup(this)
      this.setup@GlmSelectBic(AbsFilterDeg120(), uint32(524295056));
      this.scan.addShadingCorrectorBw();
    end
    
  end
  
end