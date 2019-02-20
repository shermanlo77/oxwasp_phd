classdef NullIidMeanVar < NullIid
  
  methods (Access = protected)
    
    function setup(this)
      this.setup@NullIid(uint32(4111000739));
    end
    
    function [nullMean, nullStd] = getNull(this, z)
      nullMean = mean(z);
      nullStd = std(z);
    end
    
  end
  
end

