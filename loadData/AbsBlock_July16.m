classdef AbsBlock_July16 < Scan
  
  properties
  end
  
  methods
    
    function this = AbsBlock_July16(folderLocation, fileName)
      this@Scan(folderLocation, fileName, 2000, 2000, 100, 80, 36, 708);
      
      referenceScanArray(6) = Scan();
      this.referenceScanArray = referenceScanArray;
      this.referenceScanArray(1) = Scan('data/absBlock_noFilter_July16/scans/blank_0W/', ...
          'block_black_', this.width, this.height, 20, 0, 0, this.timeExposure);
      this.referenceScanArray(2) = Scan('data/absBlock_noFilter_July16/scans/blank_10W/', ...
          'block10w_', this.width, this.height, 20, this.voltage, 10.08, this.timeExposure);
      this.referenceScanArray(3) = Scan('data/absBlock_noFilter_July16/scans/blank_18W/', ...
          'block18w_', this.width, this.height, 20, this.voltage, 18.08, this.timeExposure);
      this.referenceScanArray(4) = Scan('data/absBlock_noFilter_July16/scans/blank_28W/', ...
          'block28w_', this.width, this.height, 20, this.voltage, 28, this.timeExposure);
      this.referenceScanArray(5) = Scan('data/absBlock_noFilter_July16/scans/blank_36W/', ...
          'block36w_', this.width, this.height, 20, this.voltage, 36, this.timeExposure);
      this.referenceScanArray(6) = Scan('data/absBlock_noFilter_July16/scans/blank_44W/', ...
          'block44w_', this.width, this.height, 20, this.voltage, 44, this.timeExposure);
      
      this.referenceScanArray(2).addArtistFile('data/absBlock_noFilter_July16/sim/blank/a10.tif');
      this.referenceScanArray(3).addArtistFile('data/absBlock_noFilter_July16/sim/blank/a18.tif');
      this.referenceScanArray(4).addArtistFile('data/absBlock_noFilter_July16/sim/blank/a28.tif');
      this.referenceScanArray(5).addArtistFile('data/absBlock_noFilter_July16/sim/blank/a36.tif');
      this.referenceScanArray(6).addArtistFile('data/absBlock_noFilter_July16/sim/blank/a44.tif');
      
      this.referenceWhite = 5;
    end
    
  end
  
end

