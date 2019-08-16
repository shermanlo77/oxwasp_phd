%MIT License
%Copyright (c) 2019 Sherman Lo

classdef NullIidMixtureMedianIqr < NullIidMixture

  methods (Access = protected)
    
    function setup(this)
      this.setup@NullIidMixture(uint32(1384381930));
    end
    
    function [nullMean, nullStd] = getNull(this, z)
      q = quantile(z, [0.25, 0.5, 0.75]);
      nullMean = q(2);
      nullStd = (q(3)-q(1))/1.349;
    end
    
  end
  
end

