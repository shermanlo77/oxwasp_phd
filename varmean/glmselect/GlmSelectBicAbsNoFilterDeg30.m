%MIT License
%Copyright (c) 2019 Sherman Lo

classdef GlmSelectBicAbsNoFilterDeg30 < GlmSelectBic
  
  properties
  end
  
  methods (Access = protected)
    
    function setup(this)
      this.setup@GlmSelectBic('AbsNoFilterDeg30', uint32(4235285469));
    end
    
  end
  
end

