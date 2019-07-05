classdef GlmSelectAicAbsNoFilterNull < GlmSelectAic
  
  properties
  end
  
  methods (Access = protected)
    
    function setup(this)
      this.setup@GlmSelectAic(AbsNoFilterDeg120(), uint32(2286428987));
    end
    
  end
  
end

