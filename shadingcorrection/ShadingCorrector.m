%MIT License
%Copyright (c) 2019 Sherman Lo

%CLASS: SHADING CORRECTOR
%Shading corrects an image given an array of reference images
%Reference images are scans of the air
%Shading correction is done by using linear interpolation where y is the within image mean and x is
    %the greyvalue
%HOW TO USE:
  %pass the image size in the constructor
  %for each reference scan, pass the scan object through the method addScan(scan, index), which
      %replication to use is selected using the parameter index
  %call the method calibrate()
  %call the method shadingCorrect(scanImage) to shading correct scanImage
classdef ShadingCorrector < handle
  
  %MEMBER VARIABLES
  properties (SetAccess = protected)
    
    %array of reference images (3 dimensions)
    %dim 1 and dim 2 for the image
    %dim 3: for each image
    referenceImageArray;
    imageSize; %two vector representing the size of the image [height, width]
    nImage = 0; %number of images in referenceImageArray
    
    betweenReferenceMean; %image: between reference mean greyvalue
    globalMean; %scalar: mean of all greyvalues in referenceImageArray
    bArray; %image of the gradients

  end
  
  %METHODS
  methods (Access = public)
    
    %CONSTRUCTOR
    %PARAMETERS:
      %scan: scan object, used to extract the size of the object
    function this = ShadingCorrector(scan)
      this.imageSize = [scan.height, scan.width];
    end
    
    %METHOD: ADD SCAN
    %Add a reference scan to the shading correction
    %PARAMETERS:
      %scan: Scan object containing calibration images
      %index: integer vector, pointing to which images in that scan to use
    function addScan(this, scan, index)
      %if index is not supplied, load all images
      if nargin == 2
        referenceStack = scan.loadImageStack();
      %else load the images supplied by the parameter index
      else
        referenceStack = scan.loadImageStack(index);
      end
      %save the mean image stack of references for this stack
      this.referenceImageArray(:,:,this.nImage + 1) = mean(referenceStack, 3);
      %increment the member variable iImage
      this.nImage = this.nImage + 1;
    end
    
    %METHOD: CALIBRATE
    %Perpare statistics for shading correction
    %To be called after all reference images have been added
    function calibrate(this)
      
      %declare vector (one element for each reference image) for the within reference mean
      %this is the target greyvalue of the unshaded greyvalue for each reference image
      withinReferenceMean = zeros(1,this.nImage);
      %for each reference
      for i = 1:this.nImage
        %get the mean within image grey value and save it to withinReferenceMean
        withinReferenceMean(i) = mean(reshape(this.referenceImageArray(:,:,i),[],1));
      end
      
      %targetImageArray is a stack of this.nImage images
      %each image is completely one greyvalue, using the values in within_reference_mean
      targetImageArray = repmat(reshape(withinReferenceMean,1,1,[]),this.imageSize);
      
      %betweenReferenceMean is an image representing the between reference image mean
      this.betweenReferenceMean = mean(this.referenceImageArray,3);
      
      %globalMean is the mean of all greyvalues
      this.globalMean = mean(withinReferenceMean);
      
      %work out the sum of squares of reference image - between reference mean
      %proportional to the between reference variance
      %s_xx is an image
      sXX = sum( ...
          (this.referenceImageArray - repmat(this.betweenReferenceMean,1,1,this.nImage)).^2, ...
          3);
      %work out the covariance of between reference images and the target greyvalues
      %s_xy is an image
      sXY = sum( ...
          (this.referenceImageArray - repmat(this.betweenReferenceMean,1,1,this.nImage)) .* ...
          (targetImageArray - this.globalMean), ...
          3);
      
      %work out the gradient
      %bArray is an image
      this.bArray = sXY./sXX;
      
    end
    
    %METHOD: SHADE CORRECT
    %PARAMETERS:
    %scanImage: image to be shading corrected
    function scanImage = shadingCorrect(this, scanImage)
      %use linear interpolation for shading correction
      scanImage = this.bArray .* (scanImage - this.betweenReferenceMean) + this.globalMean;
    end
    
    %METHOD: GET NAME OF THIS SHADING CORRECTOR
    function name = getName(this)
      if this.nImage <= 2
        name = 'bw';
      else
        name = 'linear';
      end
    end
    
  end
  
end

