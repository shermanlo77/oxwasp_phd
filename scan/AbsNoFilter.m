%MIT License
%Copyright (c) 2019 Sherman Lo

classdef AbsNoFilter < Scan
  
  methods (Access = public)
    
    function this = AbsNoFilter(folderLocation, fileName)
      this@Scan(folderLocation, fileName, 2000, 2000, 100, 80, 36, 708, 6.43);
      
      calibrationScanArray(6) = Scan();
      this.calibrationScanArray = calibrationScanArray;
      this.calibrationScanArray(1) = Scan(fullfile('data','absBlock_noFilter_July16','scans', ...
          'blank_0W'), 'block_black_', this.width, this.height, 20, 0, 0, this.timeExposure, ...
          this.magnification);
      this.calibrationScanArray(2) = Scan(fullfile('data','absBlock_noFilter_July16','scans', ...
          'blank_10W'), 'block10w_', this.width, this.height, 20, this.voltage, 10.08, ...
          this.timeExposure, this.magnification);
      this.calibrationScanArray(3) = Scan(fullfile('data','absBlock_noFilter_July16','scans', ...
          'blank_18W'), 'block18w_', this.width, this.height, 20, this.voltage, 18.08, ...
          this.timeExposure, this.magnification);
      this.calibrationScanArray(4) = Scan(fullfile('data','absBlock_noFilter_July16','scans', ...
          'blank_28W'), 'block28w_', this.width, this.height, 20, this.voltage, 28, ...
          this.timeExposure, this.magnification);
      this.calibrationScanArray(5) = Scan(fullfile('data','absBlock_noFilter_July16','scans', ...
          'blank_36W'), 'block36w_', this.width, this.height, 20, this.voltage, 36, ...
          this.timeExposure, this.magnification);
      this.calibrationScanArray(6) = Scan(fullfile('data','absBlock_noFilter_July16','scans', ...
          'blank_44W'), 'block44w_', this.width, this.height, 20, this.voltage, 44, ...
          this.timeExposure, this.magnification);
      
      this.calibrationScanArray(2).addArtistFile(fullfile('data','absBlock_noFilter_July16',...
          'sim','blank','a10.tif'));
      this.calibrationScanArray(3).addArtistFile(fullfile('data','absBlock_noFilter_July16',...
          'sim','blank','a18.tif'));
      this.calibrationScanArray(4).addArtistFile(fullfile('data','absBlock_noFilter_July16',...
          'sim','blank','a28.tif'));
      this.calibrationScanArray(5).addArtistFile(fullfile('data','absBlock_noFilter_July16',...
          'sim','blank','a36.tif'));
      this.calibrationScanArray(6).addArtistFile(fullfile('data','absBlock_noFilter_July16',...
          'sim','blank','a44.tif'));
      
      this.whiteIndex = 5;
      
    end
    
  end
  
end

