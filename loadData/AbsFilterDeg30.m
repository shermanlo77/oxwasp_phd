classdef AbsFilterDeg30 < AbsFilter
  
  properties
  end
  
  methods
    
    function this = AbsFilterDeg30()
      this@AbsFilter('data/absBlock_CuFilter_Sep16/scans/phantom_30deg/', 'block30deg_');
      this.addArtistFile('data/absBlock_CuFilter_Sep16/sim/phantom/sim30.tif');
    end
    
  end
  
end

