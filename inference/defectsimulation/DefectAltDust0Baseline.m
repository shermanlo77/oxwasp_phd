%MIT License
%Copyright (c) 2019 Sherman Lo
%
%Baseline subclass does not filter the image
%Add defect BUT NO contamination
classdef DefectAltDust0Baseline < DefectAltDust0
  
  methods (Access = public)
    
    function this = DefectAltDust0Baseline()
      this@DefectAltDust0();
    end
    
  end
  
  methods (Access = protected)
    
    function setup(this)
      this.setup@DefectAltDust0(uint32(3922919431));
    end
    
    function imageFiltered = filterImage(this, imageContaminated)
      imageFiltered = imageContaminated;
    end
    
    function getFilter(this)
    end
  
  end
  
end

