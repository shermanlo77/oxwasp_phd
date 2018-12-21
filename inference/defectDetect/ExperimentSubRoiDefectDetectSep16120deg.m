classdef ExperimentSubRoiDefectDetectSep16120deg < ExperimentSubRoiDefectDetect
  
  methods (Access = public)
    
    function this = ExperimentSubRoiDefectDetectSep16120deg()
      this@ExperimentSubRoiDefectDetect('ExperimentSubRoiDefectDetectSep16120deg');
    end
    
    function printResults(this)
      this.printResults@ExperimentDefectDetect([0,5]);
    end
    
  end
  
  methods (Access = protected)
  
    function setup(this)
      this.setup@ExperimentSubRoiDefectDetect(uint32(3538096789), AbsBlock_Sep16_120deg(), ...
          [10, 50, 90, 130]);
    end
    
  end
  
end

