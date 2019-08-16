%MIT License
%Copyright (c) 2019 Sherman Lo

classdef GlmSelectBicAbsNoFilterDeg120 < GlmSelectBic
  
  properties
  end
  
  methods (Access = protected)
    
    function setup(this)
      this.setup@GlmSelectBic('AbsNoFilterDeg120', uint32(50307422));
    end
    
  end
  
end

