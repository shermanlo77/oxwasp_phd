classdef AllNullPlaneMedianIqr < AllNullPlane
  
  methods (Access = public)
    
    function this = AllNullPlaneMedianIqr()
      this@AllNullPlane();
    end
    
  end
  
  methods (Access = protected)
    
    function setup(this)
      this.setup@AllNullPlane(uint32(4112460543));
    end
    
    function filter = getFilter(this, radius)
      filter = MedianIqrNullFilter(radius);
    end
    
  end
  
end