classdef GlmSelectBicAbsNoFilterNull < GlmSelectBic
  
  properties
  end
  
  methods (Access = protected)
    
    function setup(this)
      this.setup@GlmSelectBic(AbsNoFilterDeg120(), uint32(50307422));
    end
    
  end
  
end
