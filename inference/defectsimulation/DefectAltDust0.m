%MIT License
%Copyright (c) 2019 Sherman Lo
%
%0 subclass means add defect BUT NO contamination
classdef DefectAltDust0 < DefectAlt
  
  methods (Access = public)
    
    function this = DefectAltDust0()
      this@DefectAlt();
    end
    
  end
  
  methods (Access = protected)
    
    function setup(this, seed)
      this.setup@DefectAlt(seed);
    end
    
    function defectSimulator = getDefectSimulator(this, altMean)
      %dust intensity 0.1
      defectSimulator = Dust(this.randStream, 0.1, altMean, 1);
    end
  
  end
  
end

