classdef DefectAltDust0MadMode < DefectAltDust0
  
  methods (Access = public)
    function this = DefectAltDust0MadMode()
      this@DefectAltDust0();
    end
  end
  
  methods (Access = protected)
    
    function setup(this)
      this.setup@DefectAltDust0(uint32(3890346746));
    end
    
    function filter = getFilter(this)
      filter = MadModeNullFilter(this.radius);
      filter.setSeed(this.randStream.randi([intmin('int32'),intmax('int32')],'int32'));
    end
              
  end
  
end

