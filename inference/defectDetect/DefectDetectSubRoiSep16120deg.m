classdef DefectDetectSubRoiSep16120deg < DefectDetectSubRoi
  
  methods (Access = public)
    
    function this = DefectDetectSubRoiSep16120deg()
      this@DefectDetectSubRoi();
    end
    
    function printResults(this)
      this.printResults@DefectDetectSubRoi([0,5]);
    end
    
  end
  
  methods (Access = protected)
  
    function setup(this)
      this.setup@DefectDetectSubRoi(uint32(3538096789), AbsBlock_Sep16_120deg(), ...
          [10, 50, 90, 130]);
    end
    
  end
  
end

