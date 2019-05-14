classdef TiFilterDeg120 < TiFilter
  
  properties
  end
  
  methods
    
    function this = TiFilterDeg120()
      this@TiFilter('data/titaniumBlock_SnFilter_Dec16/scans/phantom_120deg/', ...
          '120deg_');
      this.addArtistFile('data/titaniumBlock_SnFilter_Dec16/sim/phantom/120deg.tif');
    end
    
  end
  
end

