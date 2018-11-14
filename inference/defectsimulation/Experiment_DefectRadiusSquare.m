classdef Experiment_DefectRadiusSquare < Experiment_DefectRadius
  
  properties (SetAccess = private)
    defectSize = 30;
  end
  
  methods (Access = public)
    
    function this = Experiment_DefectRadiusSquare()
      this@Experiment_DefectRadius('Experiment_DefectRadiusSquare');
    end
    
  end
  
  methods (Access = protected)
    
    function setup(this)
      this.setup@Experiment_DefectRadius(uint32(3826400333));
    end
    
    function defectSimulator = getDefectSimulator(this)
      defectSimulator = PlaneMultSquare(this.randStream, this.gradContamination, ...
          this.multContamination, this.defectSize, this.altMean, this.altStd);
    end
    
  end
  
end

