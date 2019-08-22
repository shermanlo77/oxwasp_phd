classdef DefectExampleDust < DefectExample
  
  methods (Access = public)
    
    function this = DefectExampleDust()
      this@DefectExample();
    end
    
  end
  
  methods (Access = protected)
    
    %OVERRIDE: SETUP
    function setup(this)
      randStream = RandStream('mt19937ar','Seed',uint32(676943031));
      defectSimulator = PlaneMultDust(randStream, [0.01, 0.01], 2, 0.1, 3, 1);
      this.setup@DefectExample(defectSimulator, [256, 256], 20)
    end
    
  end
  
end

