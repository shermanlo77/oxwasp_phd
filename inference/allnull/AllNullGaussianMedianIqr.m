classdef AllNullGaussianMedianIqr < AllNullGaussian
  
  methods (Access = public)
    
    function this = AllNullGaussianMedianIqr()
      this@AllNullGaussian();
    end
    
  end
  
  methods (Access = protected)
    
    function setup(this)
      this.setup@AllNullGaussian(uint32(676943031));
    end
    
    function filter = getFilter(this, radius)
      filter = MedianIqrNullFilter(radius);
    end
    
  end
  
end

