%MIT License
%Copyright (c) 2020 Sherman Lo

classdef DefectDetectSubRoiTiFilterDeg30Gpu < DefectDetectSubRoiTiFilterDeg30
  
  methods (Access = public)
    
    function this = DefectDetectSubRoiTiFilterDeg30Gpu()
      this@DefectDetectSubRoiTiFilterDeg30();
    end
    
  end
  
  methods (Access = protected)
  
    %OVERRIDE to use GPU
    function filter = getFilter(this, radius)
      filter = EmpiricalNullFilterGpu(radius);
    end
    
  end
  
end

