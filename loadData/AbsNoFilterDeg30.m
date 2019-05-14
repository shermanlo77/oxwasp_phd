classdef AbsNoFilterDeg30 < AbsNoFilter
  
  properties
  end
  
  methods
    
    function this = AbsNoFilterDeg30()
      this@AbsNoFilter('data/absBlock_noFilter_July16/scans/phantom_30deg/', 'block30deg_');
      this.addArtistFile('data/absBlock_noFilter_July16/sim/phantom/sim_block30.tif');
    end
    
  end
  
end

