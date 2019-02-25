classdef DefectAltDust0Empirical < DefectAltDust0
  
  methods (Access = public)
    function this = DefectAltDust0Empirical()
      this@DefectAltDust0();
    end
  end
  
  methods (Access = protected)
    
    function setup(this)
      this.setup@DefectAltDust0(uint32(3499211588));
    end
    
    function filter = getFilter(this)
      filter = EmpiricalNullFilter(this.radius);
      filter.setSeed(this.randStream.randi([intmin('int32'),intmax('int32')],'int32'));
    end
              
  end
  
end

