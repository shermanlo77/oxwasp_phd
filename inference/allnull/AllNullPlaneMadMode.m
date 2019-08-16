%MIT License
%Copyright (c) 2019 Sherman Lo

classdef AllNullPlaneMadMode < AllNullPlane
  
  methods (Access = public)
    
    function this = AllNullPlaneMadMode()
      this@AllNullPlane();
    end
    
  end
  
  methods (Access = protected)
    
    function setup(this)
      this.setup@AllNullPlane(uint32(1196140742));
    end
    
    function filter = getFilter(this, radius)
      filter = MadModeNullFilter(radius);
      filter.setSeed(this.randStream.randi([intmin('int32'),intmax('int32')],'int32'));
    end
    
  end
  
end