classdef DefectDetectSep16120deg < DefectDetect
  
  methods (Access = public)
    
    function this = DefectDetectSep16120deg()
      this@DefectDetect();
    end
    
    function printResults(this)
      this.printResults@DefectDetect([-6, 8], [0,8], 15);
    end
    
  end
  
  methods (Access = protected)
  
    function setup(this)
      this.setup@DefectDetect(uint32(3538096789), AbsBlock_Sep16_120deg(), ...
          [10, 50, 90, 130]);
    end
    
  end
  
end

