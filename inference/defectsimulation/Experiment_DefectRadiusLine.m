classdef Experiment_DefectRadiusLine < Experiment_DefectRadius
  
  methods (Access = public)
    
    function this = Experiment_DefectRadiusLine()
      this@Experiment_DefectRadius('Experiment_DefectRadiusLine');
    end
    
  end
  
  methods (Access = protected)
    
    function setup(this)
      this.setup@Experiment_DefectRadius(uint32(4120125298));
    end
    
    function defectSimulator = getDefectSimulator(this)
      defectSimulator = PlaneMultLine(this.randStream, this.gradContamination, ...
          this.multContamination, this.altMean, this.altStd);
    end
    
  end
  
end

