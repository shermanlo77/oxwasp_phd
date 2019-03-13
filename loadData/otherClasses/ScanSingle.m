%SCAN SINGLE CLASS
%See superclass Scan
%This is a special case of Scan where there is only one image
%Overrides the method loadImage, does not have a integer suffix in the file name
classdef ScanSingle < Scan
  
  properties
  end
  
  methods
    
    %CONSTRUCTOR
    function this = ScanSingle(folderLocation, fileName, width, height, voltage, power, ...
        timeExposure)
      %calls superclass constructor
      this@Scan(folderLocation, fileName, width, height, 1, voltage, power, timeExposure);
    end
    
    %OVERRIDE: LOAD IMAGE
    %Return a sample image
    function slice = loadImage(this, ~)
      slice = imread(strcat(this.folderLocation, this.fileName,'.tif'));
      slice = this.shadingCorrect(double(slice));
    end
    
  end
  
end
