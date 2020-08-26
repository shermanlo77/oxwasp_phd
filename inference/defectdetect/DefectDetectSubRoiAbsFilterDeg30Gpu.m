%MIT License
%Copyright (c) 2020 Sherman Lo

classdef DefectDetectSubRoiAbsFilterDeg30Gpu < DefectDetectSubRoiAbsFilterDeg30
  
  methods (Access = public)
    
    function this = DefectDetectSubRoiAbsFilterDeg30Gpu()
      this@DefectDetectSubRoiAbsFilterDeg30();
    end
    
  end
  
  methods (Access = protected)
  
    %OVERRIDE to use GPU
    function filter = getFilter(this, radius)
      filter = EmpiricalNullFilterGpu(radius);
    end
    
  end
  
end

