classdef DefectExampleLine < DefectExample
  
  methods (Access = public)
    
    function this = DefectExampleLine()
      this@DefectExample();
    end
    
  end
  
  methods (Access = protected)
    
    %OVERRIDE: SETUP
    function setup(this)
      randStream = RandStream('mt19937ar','Seed',uint32(2816384857));
      defectSimulator = PlaneMultLine(randStream, [0.01, 0.01], 2, 5, 3, 1);
      this.setup@DefectExample(defectSimulator, [256, 256], 20)
    end
    
  end
  
end

