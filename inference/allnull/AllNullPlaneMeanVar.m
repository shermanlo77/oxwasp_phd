classdef AllNullPlaneMeanVar < AllNullPlane
  
  methods (Access = public)
    
    function this = AllNullPlaneMeanVar()
      this@AllNullPlane();
    end
    
  end
  
  methods (Access = protected)
    
    function setup(this)
      this.setup@AllNull(uint32(2348838239));
    end
    
    function filter = getFilter(this, radius)
      filter = MeanVarNullFilter(radius);
    end
    
  end
  
end