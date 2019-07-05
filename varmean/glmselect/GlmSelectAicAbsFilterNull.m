classdef GlmSelectAicAbsFilterNull < GlmSelectAic
  
  properties
  end
  
  methods (Access = protected)
    
    function setup(this)
      this.setup@GlmSelectAic(AbsFilterDeg120(), uint32(3361338167));
    end
    
  end
  
end

