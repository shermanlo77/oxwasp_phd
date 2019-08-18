%MIT License
%Copyright (c) 2019 Sherman Lo

classdef NullIidMixtureMeanVar < NullIidMixture
  
  methods (Access = protected)
    
    function setup(this)
      this.setup@NullIidMixture(uint32(3142929652));
    end
    
    function [nullMean, nullStd] = getNull(this, z)
      nullMean = mean(z);
      nullStd = std(z);
    end
    
  end
  
end

