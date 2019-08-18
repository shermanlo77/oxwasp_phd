%MIT License
%Copyright (c) 2019 Sherman Lo

classdef ShadingCorrectionAnovaAbsNoFilter < ShadingCorrectionAnova
  
  methods (Access = protected)
    
    function setup(this)
      this.setup@ShadingCorrectionAnova(AbsNoFilterDeg30(), uint32(680065316));
    end
    
  end
  
end
