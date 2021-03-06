%MIT License
%Copyright (c) 2019 Sherman Lo

classdef AbsFilterDeg30 < AbsFilter

  methods (Access = public)
    
    function this = AbsFilterDeg30()
      this@AbsFilter(fullfile('data','absBlock_CuFilter_Sep16','scans','phantom_30deg'), ...
          'block30deg_');
      this.addArtistFile(fullfile('data','absBlock_CuFilter_Sep16','sim','phantom','sim30.tif'));
      this.nSubSegmentation = 7;
    end
    
  end
  
end

