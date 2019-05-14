classdef TiFilterDeg120 < TiFilter
  
  methods (Access = public)
    
    function this = TiFilterDeg120()
      this@TiFilter(fullfile('data','titaniumBlock_SnFilter_Dec16','scans','phantom_120deg'), ...
          '120deg_');
      this.addArtistFile(fullfile('data','titaniumBlock_SnFilter_Dec16','sim','phantom', ...
          '120deg.tif'));
    end
    
  end
  
end

