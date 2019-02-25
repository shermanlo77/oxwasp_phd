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
      %gradient [0, 0], multiplier 1
      %dust intensity 0.1
      defectSimulator = PlaneMultDust(this.randStream, [0, 0], 1, 0.1, altMean, 1);
    end
  
  end
  
end

