%MIT License
%Copyright (c) 2019 Sherman Lo

%CLASS: PLANE MULTIPLY SQUARE
%Produces images which are Gaussian N(0,1) but a square will be N(mu, sigma^2) where the alt
    %distribution is dvia the member variables altMean and altStd. This is then multiplied by a
    %scalar followed by an addition of a plane.
classdef PlaneMultSquare < PlaneMult
  
  properties (SetAccess = protected)
    defectSize; %size of the square defect (scalar)
    %parameters of the alt distribution
    altMean;
    altStd;
  end
  
  methods (Access = public)
    
    %CONSTRUCTOR
    %PARAMETERS:
      %randStream: rng
      %grad: 2 row vector, gradient of the plane
      %multiplier: scale all pixels by this
      %defectSize: scalar, size of the square
      %altMean: alt distribution mean
      %altStd: alt distribution std
    function this = PlaneMultSquare(randStream, grad, multiplier, defectSize, altMean, altStd)
      this@PlaneMult(randStream, grad, multiplier);
      this.defectSize = defectSize;
      this.altMean = altMean;
      this.altStd = altStd;
    end
    
    %METHOD: GET DEFECTED IMAGE
    %PARAMETER:
      %size: 2 row vector [height, width]
    %RETURN:
      %image: a defected image
      %isNonNullImage: boolean map, true if that pixel is a defect
      %imagePreBias: defected Gaussian image without the smooth function added
    function [imageContaminated, isNonNullImage, imageNoContamination] = ...
          getDefectedImage(this, size)
      [imageNoContamination, isNonNullImage] = this.getDefectedImage@DefectSimulator(size);
      [imageNoContamination, isNonNullImage] = ...
          this.addSquareDefect(imageNoContamination, isNonNullImage, size/2, ...
          [this.defectSize, this.defectSize], this.altMean, this.altStd);
      imageContaminated = this.multiply(imageNoContamination, this.multiplier);
      imageContaminated = this.addPlane(imageContaminated, this.grad);
    end
    
  end
  
end

