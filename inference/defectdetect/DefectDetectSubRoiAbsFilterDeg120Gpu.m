%MIT License
%Copyright (c) 2020 Sherman Lo

classdef DefectDetectSubRoiAbsFilterDeg120Gpu < DefectDetectSubRoiAbsFilterDeg120
  
  methods (Access = public)
    
    function this = DefectDetectSubRoiAbsFilterDeg120Gpu()
      this@DefectDetectSubRoiAbsFilterDeg120();
    end
    
  end
  
  methods (Access = protected)
  
    %OVERRIDE to use GPU
    function filter = getFilter(this, radius)
      filter = EmpiricalNullFilterGpu(radius);
    end
    
  end
  
end

