classdef AbsFilterDeg120 < AbsFilter
  
  properties
    
  end
  
  methods
    
    function this = AbsFilterDeg120()
      this@AbsFilter('data/absBlock_CuFilter_Sep16/scans/phantom_120deg/', 'block120deg_');
      this.addArtistFile('data/absBlock_CuFilter_Sep16/sim/phantom/sim120.tif');
      this.nSubSegmentation = 7;
    end
    
  end
  
end

