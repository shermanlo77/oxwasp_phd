%MIT License
%Copyright (c) 2020 Sherman Lo

classdef DefectDetectSubRoiTiFilterDeg120Gpu < DefectDetectSubRoiTiFilterDeg120
  
  methods (Access = public)
    
    function this = DefectDetectSubRoiTiFilterDeg120Gpu()
      this@DefectDetectSubRoiTiFilterDeg120();
    end
    
  end
  
  methods (Access = protected)
  
    %OVERRIDE to use GPU
    function filter = getFilter(this, radius)
      filter = EmpiricalNullFilterGpu(radius);
    end
    
  end
  
end

