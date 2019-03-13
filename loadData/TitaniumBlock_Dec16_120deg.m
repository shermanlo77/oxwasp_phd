classdef TitaniumBlock_Dec16_120deg < TitaniumBlock_Dec16
  
  properties
  end
  
  methods
    
    function this = TitaniumBlock_Dec16_120deg()
      this@TitaniumBlock_Dec16('data/titaniumBlock_SnFilter_Dec16/scans/phantom_120deg/', ...
          '120deg_');
      this.addArtistFile('data/titaniumBlock_SnFilter_Dec16/sim/phantom/120deg.tif');
    end
    
  end
  
end

