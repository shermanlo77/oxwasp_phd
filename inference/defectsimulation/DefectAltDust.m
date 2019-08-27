%MIT License
%Copyright (c) 2019 Sherman Lo
%
%Add contamination and defect
classdef DefectAltDust < DefectAlt
  
  methods (Access = public)
    
    function this = DefectAltDust()
      this@DefectAlt();
    end
    
    function printResults(this)
      this.printResults@DefectAlt();
    end
    
  end
  
  methods (Access = protected)
    
    function setup(this, seed)
      this.setup@DefectAlt(seed);
    end
    
    function defectSimulator = getDefectSimulator(this, altMean)
      %gradient [0.01, 0.01], multiplier 2
      %dust intensity 0.1
      defectSimulator = PlaneMultDust(this.randStream, [0.01, 0.01], 2, 0.1, altMean, 1);
    end
  
  end
  
end

