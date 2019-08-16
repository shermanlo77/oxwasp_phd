%MIT License
%Copyright (c) 2019 Sherman Lo

classdef AbsFilterDeg120 < AbsFilter
  
  methods (Access = public)
    
    function this = AbsFilterDeg120()
      this@AbsFilter(fullfile('data','absBlock_CuFilter_Sep16','scans','phantom_120deg'), ...
          'block120deg_');
      this.addArtistFile(fullfile('data','absBlock_CuFilter_Sep16','sim','phantom','sim120.tif'));
      this.nSubSegmentation = 7;
    end
    
  end
  
end

