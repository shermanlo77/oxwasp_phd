classdef GlmSelectBicAbsNoFilterBw < GlmSelectBic
  
  properties
  end
  
  methods (Access = protected)
    
    function setup(this)
      this.setup@GlmSelectBic(AbsNoFilterDeg120(), uint32(4079019240));
      this.scan.addShadingCorrectorBw();
    end
    
  end
  
end