classdef GlmSelectAicAbsNoFilterBw < GlmSelectAic
  
  properties
  end
  
  methods (Access = protected)
    
    function setup(this)
      this.setup@GlmSelectAic(AbsNoFilterDeg120(), uint32(1489915925));
      this.scan.addShadingCorrectorBw();
    end
    
  end
  
end