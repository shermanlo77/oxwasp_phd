classdef AbsFilter < Scan
  
  methods (Access = public)
    
    function this = AbsFilter(folderLocation, fileName)
      this@Scan(folderLocation, fileName, 2000, 2000, 20, 80, 20, 500);
      
      calibrationScanArray(5) = Scan();
      this.calibrationScanArray = calibrationScanArray;
      this.calibrationScanArray(1) = Scan(fullfile('data','absBlock_CuFilter_Sep16','scans', ...
          'blank_0W'), '0w_', this.width, this.height, 20, 0, 0, this.timeExposure);
      this.calibrationScanArray(2) = Scan(fullfile('data','absBlock_CuFilter_Sep16','scans', ...
          'blank_5W'), '5w_', this.width, this.height, 20, this.voltage, 4.96, this.timeExposure);
      this.calibrationScanArray(3) = Scan(fullfile('data','absBlock_CuFilter_Sep16','scans', ...
          'blank_10W'), '10w_', this.width, this.height, 20, this.voltage, 10, this.timeExposure);
      this.calibrationScanArray(4) = Scan(fullfile('data','absBlock_CuFilter_Sep16','scans', ...
          'blank_15W'), '15w_', this.width, this.height, 20, this.voltage, 15.04, ...
          this.timeExposure);
      this.calibrationScanArray(5) = Scan(fullfile('data','absBlock_CuFilter_Sep16','scans', ...
          'blank_20W'), '20w_', this.width, this.height, 20, this.voltage, 20, this.timeExposure);
      
      this.calibrationScanArray(2).addArtistFile(...
          fullfile('data','absBlock_CuFilter_Sep16','sim','blank','a5.tif'));
      this.calibrationScanArray(3).addArtistFile(...
          fullfile('data','absBlock_CuFilter_Sep16','sim','blank','a10.tif'));
      this.calibrationScanArray(4).addArtistFile(...
          fullfile('data','absBlock_CuFilter_Sep16','sim','blank','a15.tif'));
      this.calibrationScanArray(5).addArtistFile(...
          fullfile('data','absBlock_CuFilter_Sep16','sim','blank','a20.tif'));
      
      this.whiteIndex = 5;
    end
    
  end
  
end

