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
    
    %OVERRIDE: GET BASELINE 0
    %Return same experiment but with no filter and no contaimation
    function baseline0 = getBaseline0(this)
      baseline0 = DefectAltDust0Baseline();
    end
    
    %OVERRIDE: GET BASELINE
    %Return same experiment but with no filter and with contaimation
    function baseline = getBaseline(this)
      baseline = DefectAltDustBaseline();
    end
    
    function filter = getFilter(this)
      filter = EmpiricalNullFilter(this.radius);
      filter.setSeed(this.randStream.randi([intmin('int32'),intmax('int32')],'int32'));
    end
              
  end
  
end

