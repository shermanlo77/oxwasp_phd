classdef AbsBlock_Sep16_120deg < AbsBlock_Sep16
  
  properties
    
  end
  
  methods
    
    function this = AbsBlock_Sep16_120deg()
      this@AbsBlock_Sep16('data/absBlock_CuFilter_Sep16/scans/phantom_120deg/', 'block120deg_');
      this.addArtistFile('data/absBlock_CuFilter_Sep16/sim/phantom/sim120.tif');
      this.nSubSegmentation = 7;
    end
    
  end
  
end

