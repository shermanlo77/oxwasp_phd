%MIT License
%Copyright (c) 2019 Sherman Lo

classdef DefectRadiusSquare < DefectRadius
  
  properties (SetAccess = private)
    defectSize = 30;
  end
  
  methods (Access = public)
    
    function this = DefectRadiusSquare()
      this@DefectRadius();
    end
    
  end
  
  methods (Access = protected)
    
    function setup(this)
      this.setup@DefectRadius(uint32(3826400333));
    end
    
    function defectSimulator = getDefectSimulator(this)
      defectSimulator = PlaneMultSquare(this.randStream, this.gradContamination, ...
          this.multContamination, this.defectSize, this.altMean, this.altStd);
    end
    
  end
  
end

