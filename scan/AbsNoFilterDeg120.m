classdef AbsNoFilterDeg120 < AbsNoFilter
  
  methods (Access = public)
    
    function this = AbsNoFilterDeg120()
      this@AbsNoFilter(fullfile('data','absBlock_noFilter_July16','scans','phantom_120deg'), ...
          'block120deg_');
      this.addArtistFile(fullfile('data','absBlock_noFilter_July16','sim','phantom',...
          'sim_block120.tif'));
    end
    
  end
  
end

