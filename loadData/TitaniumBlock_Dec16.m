classdef TitaniumBlock_Dec16 < Scan
  
  properties
  end
  
  methods
    
    function this = TitaniumBlock_Dec16(folderLocation, fileName)
      this@Scan(folderLocation, fileName, 2000, 2000, 20, 190, 19.95, 1415);
      
      calibrationScanArray(4) = Scan();
      this.calibrationScanArray = calibrationScanArray;
      this.calibrationScanArray(1) = Scan('data/titaniumBlock_SnFilter_Dec16/scans/blank_0W/', ...
          '0w_', this.width, this.height, 20, 0, 0, this.timeExposure);
      this.calibrationScanArray(2) = Scan('data/titaniumBlock_SnFilter_Dec16/scans/blank_10W/', ...
          '10w_', this.width, this.height, 20, this.voltage, 10.07, this.timeExposure);
      this.calibrationScanArray(3) = Scan('data/titaniumBlock_SnFilter_Dec16/scans/blank_20W/', ...
          '20w_', this.width, this.height, 20, this.voltage, 19.95, this.timeExposure);
      this.calibrationScanArray(4) = Scan('data/titaniumBlock_SnFilter_Dec16/scans/blank_26W/', ...
          '26w_', this.width, this.height, 20, this.voltage, 26.03, this.timeExposure);
      
      this.calibrationScanArray(2).addArtistFile(...
          'data/titaniumBlock_SnFilter_Dec16/sim/blank/10w.tif');
      this.calibrationScanArray(3).addArtistFile(...
          'data/titaniumBlock_SnFilter_Dec16/sim/blank/20w.tif');
      this.calibrationScanArray(4).addArtistFile(...
          'data/titaniumBlock_SnFilter_Dec16/sim/blank/26w.tif');
      
      this.whiteIndex = 3;
      
    end
    
  end
  
end

