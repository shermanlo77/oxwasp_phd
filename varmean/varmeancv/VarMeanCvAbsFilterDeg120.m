%MIT License
%Copyright (c) 2019 Sherman Lo

classdef VarMeanCvAbsFilterDeg120 < VarMeanCv
  
  methods (Access = protected)
    
    function setup(this)
      this.setup@VarMeanCv('AbsFilterDeg120', uint32(48286783));
    end
    
  end
  
end
