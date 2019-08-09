classdef DefectDetectSubRoiTiFilterDeg30 < DefectDetectSubRoi
  
  methods (Access = public)
    
    function this = DefectDetectSubRoiTiFilterDeg30()
      this@DefectDetectSubRoi();
    end
    
    function printResults(this)
      this.printResults@DefectDetectSubRoi([-10, 15], [0,15], 10);
    end
    
  end
  
  methods (Access = protected)
  
    function setup(this)
      this.setup@DefectDetectSubRoi(uint32(3437178441), TiFilterDeg30(), ...
          130);
    end
    
  end
  
end

