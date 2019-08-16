%MIT License
%Copyright (c) 2019 Sherman Lo

classdef AllNullGaussianMeanVar < AllNullGaussian
  
  methods (Access = public)
    
    function this = AllNullGaussianMeanVar()
      this@AllNullGaussian();
    end
    
  end
  
  methods (Access = protected)
    
    function setup(this)
      this.setup@AllNullGaussian(uint32(4168664255));
    end
    
    function filter = getFilter(this, radius)
      filter = MeanVarNullFilter(radius);
    end
    
  end
  
end

