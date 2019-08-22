%MIT License
%Copyright (c) 2019 Sherman Lo

classdef DefectRadiusLine < DefectRadius
  
  properties (SetAccess = private)
    lineThickness = 5;
  end
  
  methods (Access = public)
    
    function this = DefectRadiusLine()
      this@DefectRadius();
    end
    
  end
  
  methods (Access = protected)
    
    function setup(this)
      this.setup@DefectRadius(uint32(4120125298));
    end
    
    function defectSimulator = getDefectSimulator(this)
      defectSimulator = PlaneMultLine(this.randStream, this.gradContamination, ...
          this.multContamination, this.lineThickness, this.altMean, this.altStd);
    end
    
  end
  
end

