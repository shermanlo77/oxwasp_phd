%SCAN Class for handling x-ray images
classdef Scan < matlab.mixin.Heterogeneous & handle
  
  %MEMBER VARIABLES
  properties (SetAccess = protected)
    width; %width of the image
    height; %height of the image
    area; %area of the image
    nSample; %number of images
    fileName; %... name of sample image
    folderLocation; %location of the dataset
    artistFile; %location of the aRTist simulation
    referenceScanArray; %array of reference scan objects (in ascending powers)
    referenceWhite; %integer pointing to the reference for white in the phantom
    nSubSegmentation; %number of sub segmentations
    
    voltage; %in units of kV
    power; %in units of W
    timeExposure; %in units of ms
    
    panelCounter %panel counter object
    shadingCorrector; %shading corrector object
    %boolean, true to do shading correction, default false, automatically turns to true if a shading
        %corrector is added
    wantShadingCorrection;
    wantRemoveDeadPixels; %boolean, true to remove dead pixels, default false
    
    minGreyvalue; %minimum possible greyvalue
  end
  
  methods
    
    %CONSTRUCTOR
    %PARAMETERS:
      %folderLocation: location of the images
      %fileName: name of the image files
      %width: width of the image
      %height: height of the image
      %nSample: number of images
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
        
        this.wantShadingCorrection = false;
        this.wantRemoveDeadPixels = false;
        this.minGreyvalue = 0;
      end
    end
    
    %METHOD: LOAD IMAGE
    %Return a sample image
    %PARAMETERS
      %index: index of image (scalar)
    function slice = loadImage(this, index)
      slice = imread(strcat(this.folderLocation,this.fileName,num2str(index),'.tif'));
      slice = this.shadingCorrect(double(slice));
    end
    
    %METHOD: LOAD IMAGE STACK
    %Return stack of sample images
    %PARAMETERS:
      %range (optional): vector of indices of images requested
      %if empty return the full range
    function stack = loadImageStack(this, range)
      %if range not provided, provide the full range
      if nargin == 1
        range = 1:this.nSample;
      end
      %declare stack of images
      stack = zeros(this.height,this.width,numel(range));
      %for each image, put it in the stack
      for index = 1:numel(range)
        stack(:,:,index) = this.loadImage(range(index));
      end
    end
    
    %METHOD: SHADING CORRECTION
    %if wantShadingCorrection, does shading correction on the
    %provided image and returns it
    %if wantRemoveDeadPixels, remove dead pixels
    function slice = shadingCorrect(this,slice)
      if this.wantShadingCorrection
        slice = this.shadingCorrector.shadingCorrect(slice);
      end
      if this.wantRemoveDeadPixels
        slice = removeDeadPixels(slice);
      end
    end
    
    %METHOD: ADD DEFAULT SHADING CORRECTOR
    %Add bw shading corrector, using 0 power and the white power
    function addDefaultShadingCorrector(this)
      this.addShadingCorrector(ShadingCorrector(),[1,this.referenceWhite]);
    end
    
    %METHOD: ADD LINEAR SHADING CORRECTION
    %Add linear shading corrector, using 0 W and all the powers till the white power
    function addLinearShadingCorrector(this)
      this.addShadingCorrector(ShadingCorrector(),1:this.referenceWhite);
    end
    
    %METHOD: ADD SHADING CORRECTOR
    %Assign a shading corrector to the member variable and calibrate it for shading correction
    %The reference images used are determined by the parameter reference_index
    %PARAMETERS:
      %shadingCorrector: ShadingCorrector object
      %reference_index: matrix of integers, representing image index (optional), zeros are ignored
      %dim 1: #
      %dim 2: for each referenceScanArray
      %for each column, (eg 1st column for black images)
      %the mean of black images specified by dim 1 are used for shading corrector
      %
      %if not provided, use the mean of all images, black, grey and white
    function addShadingCorrector(this,shadingCorrector,referenceIndex,imageIndex)
      
      %turn off shading correction to obtain the reference images
      this.turnOffShadingCorrection();
      
      %count the number of reference scans
      n_reference = numel(referenceIndex);
      
      shadingCorrector.initalise(n_reference, this.height, this.width);
      
      %for each reference scan
      for i = 1:n_reference
        %if reference_index is not provided, take the mean of all images
        if nargin == 3
          shadingCorrector.addScan(this.referenceScanArray(referenceIndex(i)));
          %else take the mean of all images specified in the ith column of reference_index
        else
          shadingCorrector.addScan(this.referenceScanArray(referenceIndex(i)),imageIndex(:,i));
        end
      end
      
      %add the shading corrector to the member variable
      this.addManualShadingCorrector(shadingCorrector);
      
    end
    
    %METHOD: ADD MANUAL SHADING CORRECTOR
    %Assign a provided shading corrector to the member variable and calibrate it
    %for shading correction
    %PARAMETERS:
    %shadingCorrector: shadingCorrector object
    function addManualShadingCorrector(this,shadingCorrector)
      %assign the provided shading corrector to the member variable
      this.shadingCorrector = shadingCorrector;
      
      %get the minimum possible greyvalue to the shading corrector
      this.shadingCorrector.minGreyvalue = this.minGreyvalue;
      
      %if the shading corrector can smooth the reference images panel by panel
      %add the panel counter
      if this.shadingCorrector.can_smooth
        this.shadingCorrector.addPanelCounter(this.panelCounter);
      end
      
      %calibrate the shading corrector to do shading correction
      this.shadingCorrector.calibrate();
      %set shading correction to be on
      this.turnOnShadingCorrection();
      
      %add the shading corrector to each reference scan in referenceScanArray
      for i = 1:numel(this.referenceScanArray)
        this.referenceScanArray(i).shadingCorrector = this.shadingCorrector;
      end
    end
    
    %METHOD: TURN ON SHADING CORRECTION
    %Set the member variable wantShadingCorrection to be true
    function turnOnShadingCorrection(this)
      this.wantShadingCorrection = true;
      for i = 1:numel(this.referenceScanArray)
        this.referenceScanArray(i).turnOnShadingCorrection();
      end
    end
    
    %METHOD: TURN OFF SHADING CORRECTION
    %Set the memebr variable wantShadingCorrection to be false
    function turnOffShadingCorrection(this)
      this.wantShadingCorrection = false;
      for i = 1:numel(this.referenceScanArray)
        this.referenceScanArray(i).turnOffShadingCorrection();
      end
    end
    
    %METHOD: TURN ON REMOVE DEAD PIXELS
    function turnOnRemoveDeadPixels(this)
      this.wantRemoveDeadPixels = true;
      for i = 1:numel(this.referenceScanArray)
        this.referenceScanArray(i).turnOnRemoveDeadPixels();
      end
    end
    
    %METHOD: TURN OFF REMOVE DEAD PIXELS
    function turnOffRemoveDeadPixels(this)
      this.wantRemoveDeadPixels = false;
      for i = 1:numel(this.referenceScanArray)
        this.referenceScanArray(i).turnOffRemoveDeadPixels();
      end
    end
    
    %METHOD: TURN ON SET EXTREME TO NAN
    %Set the shading corrector to set extreme greyvalues to be NaN
    function turnOnSetExtremeToNan(this)
      this.shadingCorrector.turnOnSetExtremeToNan();
    end
    
    %METHOD: TURN OFF SET EXTREME TO NAN
    %Set the shading corrector to keep extreme greyvalues
    function turnOffSetExtremeToNan(this)
      this.shadingCorrector.turnOffSetExtremeToNan();
    end
    
    %METHOD: ADD ARTIST FILE
    function addArtistFile(this,artistFile)
      this.artistFile = artistFile;
    end
    
    %METHOD: GET ARTIST IMAGE
    function slice = getArtistImage(this)
      slice = double(imread(this.artistFile));
    end
    
    %METHOD: GET N REFERENCE
    function n_reference = getNReference(this)
      n_reference = numel(this.referenceScanArray);
    end
    
    %METHOD: GET POWER ARRAY
    %Return array of powers for each reference scan
    function powerArray = getPowerArray(this)
      %get the number of reference scans
      nReference = numel(this.referenceScanArray);
      %declare array of powers
      powerArray = zeros(1,nReference);
      %for each reference scan, get the power and save it
      for i = 1:n_reference
        powerArray(i) = this.referenceScanArray(i).power;
      end
    end
    
    %METHOD: GET SEGMENTATION
    %Returns a binary image, true values represent ROI
    function segmentation = getSegmentation(this)
      segmentation = this.getRoi(this.getRoiPath());
    end
    
    %METHOD: GET SUB SEGMENTATION
    %Returns a binary image for a sub segemtation, true values represent ROI
    function segmentation = getSubSegmentation(this, index)
      segmentation = this.getRoi(this.getSubRoiPath(index));
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
    
    %METHOD: GET SHADING CORRECTED ARTIST IMAGE
    %Returns the aRTist image, shading corrected
    %Uses aRTist simulations of the references except for the black image
    %PARAMETERS:
      %shadingCorrector: newly instantised shading corrector
      %referenceIndex: integer vector, pointing to which reference images to use
    %RETURN:
      %slice: shading corrected aRTist image
    function slice = getShadingCorrectedArtistImage(this, shadingCorrector, referenceIndex)
      %get the folder location and file name of the artist image
      [artistLocation,artistName,~] = fileparts(this.artistFile);
      artistLocation = strcat(artistLocation,'/');
      %instantise a Scan object containing the aRTist image
      aRTist = ScanSingle(artistLocation, artistName, this.width, this.height, this.voltage, ...
          this.power, this.timeExposure);
      %instantise an array of Scan objects, storing aRTist reference images
      %store the array in the aRTist member variable referenceScanArray
      artistReferenceArray(this.getNReference()) = Scan();
      aRTist.referenceScanArray = artistReferenceArray;
      
      referenceScan = this.referenceScanArray(1);
      greyvalue = mean(reshape(referenceScan.loadImageStack(),[],1));
      aRTist.referenceScanArray(1) = ScanSingleFlat(this.width, this.height, this.voltage, ...
          referenceScan.power, this.timeExposure, greyvalue);
      
      %for each reference scan, except for black
      for i = 2:this.getNReference()
        %get the reference scan
        referenceScan = this.referenceScanArray(i);
        %get the file location and file name of the aRTist reference image
        [artistLocation,artistName,~] = fileparts(referenceScan.artistFile);
        artistLocation = strcat(artistLocation,'/');
        %instantise a Scan object for that aRTist reference image
        aRTist.referenceScanArray(i) = ScanSingle(artistLocation, artistName, this.width, ...
            this.height, this.voltage, referenceScan.power, this.timeExposure);
      end
      
      aRTist.referenceWhite = this.referenceWhite;
      %add shading correction and get the shading corrected aRTist image
      aRTist.addShadingCorrector(shadingCorrector,referenceIndex);
      slice = aRTist.loadImage();
    end
    
  end
  
end

