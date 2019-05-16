%SCAN Class for handling x-ray images
%matlab.mixin.Heterogeneous allows for subclasses to be stored in an array
classdef Scan < matlab.mixin.Heterogeneous & handle
  
  %MEMBER VARIABLES
  properties (SetAccess = protected)
    width; %width of the image
    height; %height of the image
    area; %area of the image
    nSample; %number of images or replications
    fileName; %name of sample image
    folderLocation; %location of the dataset
    artistFile; %location of the aRTist simulation
    calibrationScanArray; %array of calibration scan objects (in ascending powers)
    whiteIndex; %integer pointing to the calibration for white in the phantom
    nSubSegmentation; %number of sub segmentations
    
    voltage; %in units of kV
    power; %in units of W
    timeExposure; %in units of ms
    
    panelCounter %panel counter object
    shadingCorrector; %shading corrector object
    %boolean, true to do shading correction, default false, automatically turns to true if a shading
        %corrector is added
    wantShadingCorrection = false;
    
  end
  
  methods (Access = public)
    
    %CONSTRUCTOR
    %PARAMETERS:
      %folderLocation: location of the images
      %fileName: name of the image files (appended with a sequential number for each replication)
      %width: width of the image
      %height: height of the image
      %nSample: number of images or replications
      %voltage: in units of kV
      %power: in units of W
      %timeExposure: in units of ms
    function this = Scan(folderLocation, fileName, width, height, nSample, voltage, power, ...
        timeExposure)
      %assign member variable if parameters are provided
      if nargin > 0
        this.folderLocation = folderLocation;
        this.fileName = fileName;
        this.width = width;
        this.height = height;
        this.area = width * height;
        this.nSample = nSample;
        this.voltage = voltage;
        this.power = power;
        this.timeExposure = timeExposure;
      end
    end
    
    %METHOD: LOAD IMAGE
    %Return a sample image
    %PARAMETERS
      %index: index of image, integer
    %RETURN;
      %slice: height x width image
    function slice = loadImage(this, index)
      %file name is appended with the index
      slice = imread( fullfile( ...
          this.folderLocation, strcat( this.fileName, num2str(index), '.tif') ) );
      slice = this.shadingCorrect(double(slice));
    end
    
    %METHOD: LOAD IMAGE STACK
    %Return stack of sample images
    %PARAMETERS:
      %range (optional): vector of indices of images requested, if empty return the full range
    %RETURN:
      %stack: height x width x numel(range), contains image for each index in range
    function stack = loadImageStack(this, range)
      %if range not provided, provide the full range
      if nargin == 1
        range = 1:this.nSample;
      end
      %declare stack of images
      stack = zeros(this.height, this.width, numel(range));
      %for each image, put it in the stack
      for index = 1:numel(range)
        stack(:,:,index) = this.loadImage(range(index));
      end
    end
    
    %METHOD: SHADING CORRECTION
    %if wantShadingCorrection, does shading correction on the provided image and returns it
    function slice = shadingCorrect(this, slice)
      if this.wantShadingCorrection
        slice = this.shadingCorrector.shadingCorrect(slice);
      end
    end
    
    %METHOD: ADD BLACK WHITE SHADING CORRECTOR
    %Add bw shading corrector, using 0 power and the white power
    %PARAMETERS:
      %imageIndex (optional): 2 column matrix of integers, size # x 2
          %dim 1: points to which replication to use
          %dim 2: [black, white]
          %if imageIndex is not provided, then all replications are used
    function addShadingCorrectorBw(this, imageIndex)
      if (nargin == 1)
        this.addShadingCorrector(ShadingCorrector(this), [1,this.whiteIndex]);
      else
        this.addShadingCorrector(ShadingCorrector(this), [1,this.whiteIndex], imageIndex);
      end
    end
    
    %METHOD: ADD LINEAR SHADING CORRECTION
    %Add linear shading corrector, using 0 W and all the powers till the white power
    %PARAMETERS:
      %calibrationIndex (optional): vector of integers, which currents to use
      %imageIndex (optional): matrix of integers, size # x numel(calibrationIndex)
        %dim 1: for the current in calibrationIndex(#), points to which replication to use
        %dim 2: for each current specified in calibrationIndex
      %if calibrationIndex and imageIndex is not provided, then all replications are used from black
          %to white
    function addShadingCorrectorLinear(this, calibrationIndex, imageIndex)
      if (nargin == 1)
        this.addShadingCorrector(ShadingCorrector(this), 1:this.whiteIndex);
      else
        this.addShadingCorrector(ShadingCorrector(this), calibrationIndex, imageIndex);
      end
    end
    
    %METHOD: ADD SHADING CORRECTOR
    %Assign a shading corrector to the member variable and calibrate it for shading correction
    %Which calibration images to use for the shading correction is specified in the parameter
    %PARAMETERS:
      %shadingCorrector: ShadingCorrector object
      %calibrationIndex: vector of integers, which currents to use
      %imageIndex (optional): matrix of integers, size # x numel(calibrationIndex)
        %dim 1: for the current in calibrationIndex(#), points to which replication to use
        %dim 2: for each current specified in calibrationIndex
        %if imageIndex is not provided, then all replications are used
    function addShadingCorrector(this, shadingCorrector, calibrationIndex, imageIndex)
      
      %turn off shading correction to obtain the calibration images
      this.setShadingCorrectionOff();
      
      %count the number of calibration scans
      nCalibration = numel(calibrationIndex);
      
      %for each calibration scan
      for i = 1:nCalibration
        %if imageIndex is not provided, take the mean of all images
        if nargin == 3
          shadingCorrector.addScan(this.calibrationScanArray(calibrationIndex(i)));
        %else take the mean of all images specified in the ith column of imageIndex
        else
          shadingCorrector.addScan(this.calibrationScanArray(calibrationIndex(i)),imageIndex(:,i));
        end
      end
      %calibrate the shading corrector and add it to the member variable
      shadingCorrector.calibrate();
      this.addShadingCorrectorManual(shadingCorrector);
      
    end
    
    %METHOD: ADD MANUAL SHADING CORRECTOR
    %Assign a provided calibrated shading corrector to the member variable and also to all
        %calibration images
    %IMPORTANT: the provided shadingCorrector should be calibrated using the method calibrate()
    %IMPORANT: Each calibration has a shallow copy of the provided shadingCorrector
    %PARAMETERS:
      %shadingCorrector: shadingCorrector object, calibration images should be provided to it
    function addShadingCorrectorManual(this, shadingCorrector)
      %assign the provided shading corrector to the member variable
      this.shadingCorrector = shadingCorrector;
      %set shading correction to be on
      this.setShadingCorrectionOn();
      %add the shading corrector to each calibration scan in calibrationScanArray, shallow copies
      for i = 1:numel(this.calibrationScanArray)
        this.calibrationScanArray(i).addShadingCorrectorManual(shadingCorrector);
      end
      
    end
    
    %METHOD: SET SHADING CORRECTION ON
    %Set the member variable wantShadingCorrection for this and all calibration images to be true
    function setShadingCorrectionOn(this)
      this.wantShadingCorrection = true;
      for i = 1:numel(this.calibrationScanArray)
        this.calibrationScanArray(i).setShadingCorrectionOn();
      end
    end
    
    %METHOD: SET SHADING CORRECTION OFF
    %Set the member variable wantShadingCorrection for this and all calibration images to be false
    function setShadingCorrectionOff(this)
      this.wantShadingCorrection = false;
      for i = 1:numel(this.calibrationScanArray)
        this.calibrationScanArray(i).setShadingCorrectionOff();
      end
    end

    %METHOD: GET ARTIST IMAGE
    function slice = getArtistImage(this)
      slice = double(imread(this.artistFile));
    end
    
    %METHOD: GET N CALIBRATION
    function calibration = getNCalibration(this)
      calibration = numel(this.calibrationScanArray);
    end
    
    %METHOD: GET POWER ARRAY
    %Return array of powers (W) for each calibration scan
    function powerArray = getPowerArray(this)
      %get the number of calibration scans
      nCalibration = this.getNCalibration();
      %declare array of powers
      powerArray = zeros(1, nCalibration);
      %for each calibration scan, get the power and save it
      for i = 1:nCalibration
        powerArray(i) = this.calibrationScanArray(i).power;
      end
    end
    
    %METHOD: GET SEGMENTATION
    %Returns a binary image, true values represent ROI
    function segmentation = getSegmentation(this)
      segmentation = this.getRoi(this.getRoiPath());
    end
    
    %METHOD: GET SUB SEGMENTATION
    %Returns a binary image for a sub segemtation, true values represent ROI
    %PARAMETER:
      %index: integer, for each sub segmentation
    function segmentation = getSubSegmentation(this, index)
      segmentation = this.getRoi(this.getSubRoiPath(index));
    end

    %METHOD: GET SHADING CORRECTED ARTIST IMAGE
    %Returns the aRTist image, shading corrected
    %Shading correction is done using aRTist simulations of the calibrations except for the black
        %image because this is not provided. Instead a flat image using the mean of the black images
        %is used for the black image in calibrating the shading correction
    %PARAMETERS:
      %shadingCorrectorClass: class name of the shading corrector
      %calibrationIndex: integer vector, pointing to which calibration images to use
    %RETURN:
      %slice: shading corrected aRTist image
    function slice = getArtistImageShadingCorrected(this, shadingCorrectorClass, calibrationIndex)
      %instantise shading corrector
      shadingCorrectorArtist = feval(shadingCorrectorClass, this);
      %get the folder location and file name of the artist image
      [artistLocation, artistName, ~] = fileparts(this.artistFile);
      %instantise a Scan object containing the aRTist image
      aRTist = ScanSingle(artistLocation, artistName, this.width, this.height, this.voltage, ...
          this.power, this.timeExposure);
      %instantise an array of Scan objects, storing aRTist calibration images
      %store the array in the aRTist member variable calibrationScanArray
      artistCalibrationArray(this.getNCalibration()) = Scan();
      aRTist.calibrationScanArray = artistCalibrationArray;
      
      %use the mean of the black image for the artist simulation of the black image
      calibrationScan = this.calibrationScanArray(1);
      greyvalue = mean(reshape(calibrationScan.loadImageStack(),[],1));
      aRTist.calibrationScanArray(1) = ScanSingleFlat(this.width, this.height, this.voltage, ...
          calibrationScan.power, this.timeExposure, greyvalue);
      
      %for each calibration scan, except for black
      for i = 2:this.getNCalibration()
        %get the calibration scan
        calibrationScan = this.calibrationScanArray(i);
        %get the file location and file name of the aRTist calibration image
        [artistLocation, artistName, ~] = fileparts(calibrationScan.artistFile);
        %instantise a Scan object for that aRTist calibration image
        aRTist.calibrationScanArray(i) = ScanSingle(artistLocation, artistName, this.width, ...
            this.height, this.voltage, calibrationScan.power, this.timeExposure);
      end
      
      aRTist.whiteIndex = this.whiteIndex;
      %add shading correction and get the shading corrected aRTist image
      aRTist.addShadingCorrector(shadingCorrectorArtist, calibrationIndex);
      slice = aRTist.loadImage();
    end
    
    %METHOD: GET SHADING CORRECTOR STATUS
    %Returns a string describing the shading corrector
    function name = getShadingCorrectionStatus(this)
      if (this.wantShadingCorrection)
        name = this.shadingCorrector.getName();
      else
        name = 'null';
      end
    end
    
  end
  
  methods (Access = protected)
    
    %METHOD: ADD ARTIST FILE
    function addArtistFile(this, artistFile)
      this.artistFile = artistFile;
    end
    
    %METHOD: GET ROI
    %Returns the mask of a roi in the .roi file specified in roiPath
    function segmentation = getRoi(this, roiPath)
      opener = ij.io.Opener();
      roi = opener.openRoi(roiPath);
      
      %get the coordinates of the roi
      %note: matlab starts at 1, java starts at 0
      roiRectangle = roi.getBounds();
      x = roiRectangle.x + 1;
      y = roiRectangle.y + 1;
      
      %get the mask of the roi, this returns an imageProcessor which represent the roi with a
      %non-zero value, this image is also cropped
      mask = roi.getMask();
      %copy the values from java to matlab
      roiMask = zeros(mask.getWidth(), mask.getHeight());
      roiMask(1:end) = mask.getPixels();
      roiMask = logical(-roiMask');
      
      %copy the pixels from the mask to the segmentation matrix at the roi coordinates
      segmentation = false(this.height, this.width);
      segmentation(y:(y+mask.getHeight()-1), x:(x+mask.getWidth()-1)) = roiMask;
    end
    
    %METHOD: GET ROI PATH
    %Returns the path of the region of interst file
    function roiPath = getRoiPath(this)
      roiPath = strcat(this.folderLocation,'segmentation.roi');
    end
    
    %METHOD: GET SUB ROI PATH
    %Returns the path of the sub region of interst file
    function roiPath = getSubRoiPath(this, index)
      roiPath = strcat(this.folderLocation,'segmentation',num2str(index),'.roi');
    end
    
  end
  
end

