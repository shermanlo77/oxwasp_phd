%MIT License
%Copyright (c) 2019 Sherman Lo

classdef TiFilter < Scan
  
  methods (Access = public)
    
    function this = TiFilter(folderLocation, fileName)
      this@Scan(folderLocation, fileName, 2000, 2000, 20, 190, 19.95, 1415, 11.2);
      
      calibrationScanArray(4) = Scan();
      this.calibrationScanArray = calibrationScanArray;
      this.calibrationScanArray(1) = Scan(fullfile('data','titaniumBlock_SnFilter_Dec16', ...
          'scans','blank_0W'),'0w_', this.width, this.height, 20, 0, 0, this.timeExposure, ...
          this.magnification);
      this.calibrationScanArray(2) = Scan(fullfile('data','titaniumBlock_SnFilter_Dec16', ...
          'scans','blank_10W'),'10w_', this.width, this.height, 20, this.voltage, 10.07, ...
          this.timeExposure, this.magnification);
      this.calibrationScanArray(3) = Scan(fullfile('data','titaniumBlock_SnFilter_Dec16', ...
          'scans','blank_20W'),'20w_', this.width, this.height, 20, this.voltage, 19.95, ...
          this.timeExposure, this.magnification);
      this.calibrationScanArray(4) = Scan(fullfile('data','titaniumBlock_SnFilter_Dec16', ...
          'scans','blank_26W'),'26w_', this.width, this.height, 20, this.voltage, 26.03, ...
          this.timeExposure, this.magnification);
      
      this.calibrationScanArray(2).addArtistFile(...
          fullfile('data','titaniumBlock_SnFilter_Dec16','sim','blank','10w.tif'));
      this.calibrationScanArray(3).addArtistFile(...
          fullfile('data','titaniumBlock_SnFilter_Dec16','sim','blank','20w.tif'));
      this.calibrationScanArray(4).addArtistFile(...
          fullfile('data','titaniumBlock_SnFilter_Dec16','sim','blank','26w.tif'));
      
      this.whiteIndex = 3;
      
    end
    
  end
  
end

