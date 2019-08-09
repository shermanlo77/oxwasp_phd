classdef DefectDetectSubRoiTiFilterDeg120 < DefectDetectSubRoi
  
  methods (Access = public)
    
    function this = DefectDetectSubRoiTiFilterDeg120()
      this@DefectDetectSubRoi();
    end
    
    function printResults(this)
      this.printResults@DefectDetectSubRoi([-10, 15], [0,15], 10);
    end
    
  end
  
  methods (Access = protected)
  
    function setup(this)
      this.setup@DefectDetectSubRoi(uint32(609397184), TiFilterDeg120(), ...
          130);
    end
    
  end
  
end

