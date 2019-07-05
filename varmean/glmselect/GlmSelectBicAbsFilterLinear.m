classdef GlmSelectBicAbsFilterLinear < GlmSelectBic
  
  properties
  end
  
  methods (Access = protected)
    
    function setup(this)
      this.setup@GlmSelectBic(AbsFilterDeg120(), uint32(646525939));
      this.scan.addShadingCorrectorLinear();
    end
    
  end
  
end