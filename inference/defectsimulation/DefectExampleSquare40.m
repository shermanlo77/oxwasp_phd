classdef DefectExampleSquare40 < DefectExample
  
  methods (Access = public)
    
    function this = DefectExampleSquare40()
      this@DefectExample();
    end
    
  end
  
  methods (Access = protected)
    
    %OVERRIDE: SETUP
    %kernel radius 40
    function setup(this)
      randStream = RandStream('mt19937ar','Seed',uint32(4120988592));
      defectSimulator = PlaneMultSquare(randStream, [0.01, 0.01], 2, 30, 3, 1);
      this.setup@DefectExample(defectSimulator, [256, 256], 40)
    end
    
  end
  
end

