classdef AbsBlock_Mar16 < Scan
  %BLOCKDATA
  %Class for obtaining images for the 140316 data
  
  %MEMBER VARIABLES
  properties
    
  end
  
  %METHODS
  methods
    
    %CONSTRUCTOR
    function this = AbsBlock_Mar16()
      this@Scan('data/absBlock_Mar16/', 'block_', 1996, 1996, 100, 100, 33, 500);
      %assign member variables
      this.panel_counter = PanelCounter_Brass();
      %this.min_greyvalue = 5.7588E3;
    end
    
    %OVERRIDE: SEGMENTATION ONLY TOP HALF
    function segmentation = getSegmentation(this)
      segmentation = this.getSegmentation@Scan();
      segmentation((floor(this.height/2)+1):end,:) = false;
    end
    
    
  end
  
end
