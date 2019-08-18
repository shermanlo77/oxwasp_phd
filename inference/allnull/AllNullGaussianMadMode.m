%MIT License
%Copyright (c) 2019 Sherman Lo

classdef AllNullGaussianMadMode < AllNullGaussian

  methods (Access = public)
    
    function this = AllNullGaussianMadMode()
      this@AllNullGaussian();
    end
    
  end
  
  methods (Access = protected)
    
    function setup(this)
      this.setup@AllNullGaussian(uint32(4144164702));
    end
    
    function filter = getFilter(this, radius)
      filter = MadModeNullFilter(radius);
      filter.setSeed(this.randStream.randi([intmin('int32'),intmax('int32')],'int32'));
    end
    
  end
  
end

