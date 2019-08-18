%MIT License
%Copyright (c) 2019 Sherman Lo

classdef DefectDetectSubRoiAbsFilterDeg120 < DefectDetectSubRoi
  
  methods (Access = public)
    
    function this = DefectDetectSubRoiAbsFilterDeg120()
      this@DefectDetectSubRoi();
    end
    
    function printResults(this)
      this.printResults@DefectDetectSubRoi([-6, 8], [0,8], 15);
    end
    
  end
  
  methods (Access = protected)
  
    function setup(this)
      this.setup@DefectDetectSubRoi(uint32(3538096789), AbsFilterDeg120(), ...
          [10, 50, 90, 130]);
    end
    
  end
  
end

