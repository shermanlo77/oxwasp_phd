classdef GlmSelectAic < GlmSelect
  
  properties
  end
  
  methods (Access = protected)
    
    function setup(this, scan, seed)
      this.setup@GlmSelect(scan, seed);
    end
    
    function aic = getCriterion(this, glm)
      aic = 2*glm.NumCoefficients - 2*this.getLogLikelihood(glm);
    
  end
  
end

