%MIT License
%Copyright (c) 2019 Sherman Lo

classdef AbsNoFilterDeg30 < AbsNoFilter
  
  methods (Access = public)
    
    function this = AbsNoFilterDeg30()
      this@AbsNoFilter(fullfile('data','absBlock_noFilter_July16','scans','phantom_30deg'), ...
          'block30deg_');
      this.addArtistFile(fullfile('data','absBlock_noFilter_July16','sim','phantom',...
          'sim_block30.tif'));
    end
    
  end
  
end

