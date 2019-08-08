classdef TiFilterDeg30 < TiFilter
  
  methods (Access = public)
    
    function this = TiFilterDeg30()
      this@TiFilter(fullfile('data','titaniumBlock_SnFilter_Dec16','scans','phantom_30deg'), ...
          '30deg_');
      this.addArtistFile(fullfile('data','titaniumBlock_SnFilter_Dec16','sim','phantom', ...
          '30deg.tif'));
      this.nSubSegmentation = 7;
    end
    
  end
  
end

