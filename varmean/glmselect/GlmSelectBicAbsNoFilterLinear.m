classdef GlmSelectBicAbsNoFilterLinear < GlmSelectBic
  
  properties
  end
  
  methods (Access = protected)
    
    function setup(this)
      this.setup@GlmSelectBic(AbsNoFilterDeg120(), uint32(533981373));
      this.scan.addShadingCorrectorLinear();
    end
    
  end
  
end