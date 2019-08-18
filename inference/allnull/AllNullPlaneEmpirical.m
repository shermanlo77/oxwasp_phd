%MIT License
%Copyright (c) 2019 Sherman Lo

classdef AllNullPlaneEmpirical < AllNullPlane
  
  methods (Access = public)
    
    function this = AllNullPlaneEmpirical()
      this@AllNullPlane();
    end
    
  end
  
  methods (Access = protected)
    
    function setup(this)
      this.setup@AllNullPlane(uint32(2084672537));
    end
    
    function filter = getFilter(this, radius)
      filter = EmpiricalNullFilter(radius);
      filter.setSeed(this.randStream.randi([intmin('int32'),intmax('int32')],'int32'));
    end
    
  end
  
end

