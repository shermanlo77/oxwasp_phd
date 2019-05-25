classdef Bgw_Mar16 < Scan
  
  properties
  end
  
  methods
    
    function this = Bgw_Mar16()
      this@Scan([], [], 1996, 1996, 100, 85, 6.8, 1000);
      %assign member variables
      this.panelCounter = PanelCounterBrass();
      
      this.whiteIndex = 3;
      calibrationScanArray(3) = Scan();
      this.calibrationScanArray = calibrationScanArray;
      this.calibrationScanArray(1) = Scan('data/bgw_Mar16/black/', 'black_140316_', ...
          this.width, this.height, 20, 0, 0, 1000);
      this.calibrationScanArray(2) = Scan('data/bgw_Mar16/grey/', 'grey_140316_', ...
          this.width, this.height, 20, 85, 1.7, 1000);
      this.calibrationScanArray(3) = Scan('data/bgw_Mar16/white/', 'white_140316_', ...
          this.width, this.height, 20, 85, 6.8, 1000);
    end
  end
  
end
