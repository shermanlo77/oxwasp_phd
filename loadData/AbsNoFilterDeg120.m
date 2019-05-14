classdef AbsNoFilterDeg120 < AbsNoFilter
  
  properties
  end
  
  methods
    
    function this = AbsNoFilterDeg120()
      this@AbsNoFilter('data/absBlock_noFilter_July16/scans/phantom_120deg/', 'block120deg_');
      this.addArtistFile('data/absBlock_noFilter_July16/sim/phantom/sim_block120.tif');
    end
    
  end
  
end

