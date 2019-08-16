%MIT License
%Copyright (c) 2019 Sherman Lo

classdef VarMeanCvAbsFilterDeg30 < VarMeanCv
  
  methods (Access = protected)
    
    function setup(this)
      this.setup@VarMeanCv('AbsFilterDeg30', uint32(287531990));
    end
    
  end
  
end
