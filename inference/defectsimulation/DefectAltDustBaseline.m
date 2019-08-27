%MIT License
%Copyright (c) 2019 Sherman Lo
%
%Baseline subclass does not filter the image
%Add contamination and defect
classdef DefectAltDustBaseline < DefectAltDust
  
  methods (Access = public)
    
    function this = DefectAltDustBaseline()
      this@DefectAltDust();
    end
    
  end
  
  methods (Access = protected)
    
    function setup(this)
      this.setup@DefectAltDust(uint32(545404223));
    end
    
    function imageFiltered = filterImage(this, imageContaminated)
      imageFiltered = imageContaminated;
    end
    
    function getFilter(this)
    end
  
  end
  
end

