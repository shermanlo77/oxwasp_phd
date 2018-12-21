classdef ExperimentDefectDetectSep16120deg < ExperimentDefectDetect
  
  methods (Access = public)
    
    function this = ExperimentDefectDetectSep16120deg()
      this@ExperimentDefectDetect('ExperimentDefectDetectSep16120deg');
    end
    
    function printResults(this)
      this.printResults@ExperimentDefectDetect([0,5]);
    end
    
  end
  
  methods (Access = protected)
  
    function setup(this)
      this.setup@ExperimentDefectDetect(uint32(3538096789), AbsBlock_Sep16_120deg(), ...
          [10, 50, 90, 130]);
    end
    
  end
  
end

