classdef ShadingCorrectionAnovaAbsFilter < ShadingCorrectionAnova
  
  methods (Access = protected)
    
    function setup(this)
      this.setup@ShadingCorrectionAnova(AbsFilterDeg30(), uint32(2173068990));
    end
    
  end
  
end
