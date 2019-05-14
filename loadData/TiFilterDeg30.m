classdef TiFilterDeg30 < TiFilter
  
  properties
  end
  
  methods
    
    function this = TiFilterDeg30()
      this@TiFilter('data/titaniumBlock_SnFilter_Dec16/scans/phantom_30deg/', '30deg_');
      this.addArtistFile('data/titaniumBlock_SnFilter_Dec16/sim/phantom/30deg.tif');
    end
    
  end
  
end

