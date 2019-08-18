%MIT License
%Copyright (c) 2019 Sherman Lo

classdef DefectAltDustEmpirical < DefectAltDust
  
  methods (Access = public)
    function this = DefectAltDustEmpirical()
      this@DefectAltDust();
    end
  end
  
  methods (Access = protected)
    
    function setup(this)
      this.setup@DefectAltDust(uint32(153380491));
    end
    
    function filter = getFilter(this)
      filter = EmpiricalNullFilter(this.radius);
      filter.setSeed(this.randStream.randi([intmin('int32'),intmax('int32')],'int32'));
    end
              
  end
  
end

