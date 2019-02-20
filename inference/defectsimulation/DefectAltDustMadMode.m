classdef DefectAltDustMadMode < DefectAltDust
  
  methods (Access = public)
    function this = DefectAltDustMadMode()
      this@DefectAltDust();
    end
  end
  
  methods (Access = protected)
    
    function setup(this)
      this.setup@DefectAltDust(uint32(3437178441));
    end
    
    function filter = getFilter(this)
      filter = EmpiricalNullFilter(this.radius);
      filter.setSeed(this.randStream.randi([intmin('int32'),intmax('int32')],'int32'));
    end
              
  end
  
end

