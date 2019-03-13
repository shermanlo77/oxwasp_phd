classdef AbsBlock_July16_120deg < AbsBlock_July16
  
  properties
  end
  
  methods
    
    function this = AbsBlock_July16_120deg()
      this@AbsBlock_July16('data/absBlock_noFilter_July16/scans/phantom_120deg/', 'block120deg_');
      this.addArtistFile('data/absBlock_noFilter_July16/sim/phantom/sim_block120.tif');
    end
    
  end
  
end

