%MIT License
%Copyright (c) 2019 Sherman Lo

classdef AllNullGaussianEmpirical < AllNullGaussian
  
  methods (Access = public)
    
    function this = AllNullGaussianEmpirical()
      this@AllNullGaussian();
    end
    
  end
  
  methods (Access = protected)
    
    function setup(this)
      this.setup@AllNullGaussian(uint32(3499211588));
    end
    
    function filter = getFilter(this, radius)
      filter = EmpiricalNullFilter(radius);
      filter.setSeed(this.randStream.randi([intmin('int32'),intmax('int32')],'int32'));
    end
    
  end
  
end

