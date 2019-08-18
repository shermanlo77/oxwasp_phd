%MIT License
%Copyright (c) 2019 Sherman Lo

%CLASS: PLANE MULTIPLY LINE
%Produces images which are Gaussian N(0,1) but a verticle line will be N(mu, sigma^2) where the alt
    %distribution is dvia the member variables altMean and altStd. This is then multiplied by a
    %scalar followed by an addition of a plane.
classdef PlaneMultLine < PlaneMult
  
  properties (SetAccess = protected)
    %parameters of the alt distribution
    altMean;
    altStd;
    lineThickness; %how thick the defect line is
  end
  
  methods (Access = public)
    
    %CONSTRUCTOR
    %PARAMETERS:
      %randStream: rng
      %grad: 2 row vector, gradient of the plane
      %multiplier: scale all pixels by this
      %altMean: alt distribution mean
      %altStd: alt distribution std
      %lineThickness: how thick the defect line is
    function this = PlaneMultLine(randStream, grad, multiplier, altMean, altStd, lineThickness)
      this@PlaneMult(randStream, grad, multiplier);
      this.altMean = altMean;
      this.altStd = altStd;
      this.lineThickness = lineThickness;
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
          this.addLineDefect(imageNoContamination, isNonNullImage, round(size(2)/2), ...
          this.lineThickness, this.altMean, this.altStd);
      imageContaminated = this.multiply(imageNoContamination, this.multiplier);
      imageContaminated = this.addPlane(imageContaminated, this.grad);
    end
    
  end
  
end

