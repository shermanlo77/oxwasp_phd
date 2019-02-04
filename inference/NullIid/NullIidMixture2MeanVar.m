classdef NullIidMixture2MeanVar < NullIidMixture2
  
  methods (Access = protected)
    
    function setup(this)
      this.setup@NullIidMixture2(uint32(830551256));
    end
    
    function [nullMean, nullStd] = getNull(this, z)
      nullMean = mean(z);
      nullStd = std(z);
    end
    
  end
  
end

