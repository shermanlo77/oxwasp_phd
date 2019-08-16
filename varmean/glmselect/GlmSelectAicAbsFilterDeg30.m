%MIT License
%Copyright (c) 2019 Sherman Lo

classdef GlmSelectAicAbsFilterDeg30 < GlmSelectAic
  
  properties
  end
  
  methods (Access = protected)
    
    function setup(this)
      this.setup@GlmSelectAic('AbsFilterDeg30', uint32(1939773748));
    end
    
  end
  
end

