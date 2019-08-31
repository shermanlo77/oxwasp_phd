%MIT License
%Copyright (c) 2019 Sherman Lo

classdef DefectDetectSubRoiAbsFilterDeg30 < DefectDetectSubRoi
  
  methods (Access = public)
    
    function this = DefectDetectSubRoiAbsFilterDeg30()
      this@DefectDetectSubRoi();
    end
    
    function printResults(this)
      this.printResults@DefectDetectSubRoi([-6, 8], [0,8], 15, 1);
    end
    
  end
  
  methods (Access = protected)
  
    function setup(this)
      this.setup@DefectDetectSubRoi(uint32(2084672537), AbsFilterDeg30(), ...
          130);
    end
    
  end
  
end

