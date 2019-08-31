%MIT License
%Copyright (c) 2019 Sherman Lo

classdef DefectDetectAbsFilterDeg120 < DefectDetect
  
  methods (Access = public)
    
    function this = DefectDetectAbsFilterDeg120()
      this@DefectDetect();
    end
    
    function printResults(this)
      this.printResults@DefectDetect([-6, 8], [0,8], 15, 1);
    end
    
  end
  
  methods (Access = protected)
  
    function setup(this)
      this.setup@DefectDetect(uint32(3538096789), AbsFilterDeg120(), ...
          [10, 50, 90, 130]);
    end
    
  end
  
end

