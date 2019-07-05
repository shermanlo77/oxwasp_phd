classdef GlmSelectBic < GlmSelect
  
  properties
  end
  
  methods (Access = protected)
    
    function setup(this, scan, seed)
      this.setup@GlmSelect(scan, seed);
    end
    
    function bic = getCriterion(this, glm)
      bic = log(glm.NumObservations)*glm.NumCoefficients - 2*glm.LogLikelihood;
    end
    
  end
  
end