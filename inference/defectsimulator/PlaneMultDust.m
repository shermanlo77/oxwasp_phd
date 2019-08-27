%MIT License
%Copyright (c) 2019 Sherman Lo

%CLASS: PLANE MULTIPLY DUST
%Produces images which are Gaussian (1-p)N(0,1) + pN(mu, sigma^2) where the alt distribution is 
    %described via the member variables altMean and altStd. This is then multiplied by a scalar
    %followed by an addition of a plane.
classdef PlaneMultDust < PlaneMult
  
  properties (SetAccess = protected)
    p; %probability of alt
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
      %p: probability pixel is alt
      %altMean: alt distribution mean
      %altStd: alt distribution std
    function this = PlaneMultDust(randStream, grad, multiplier, p, altMean, altStd)
      this@PlaneMult(randStream, grad, multiplier);
      this.p = p;
      this.altMean = altMean;
      this.altStd = altStd;
    end
    
    %METHOD: GET DEFECTED IMAGE
    %Return an image with N(0,1) or alt distributed pixels x multiplier + plane
    %PARAMETER:
      %size: 2 row vector [height, width]
    %RETURN:
      %image: a defected Gaussian image
      %isNonNullImage: boolean map, true if that pixel is a defect
      %imageNoContamination: image with defect but no contamination
    function [imageContaminated, isNonNullImage, imageNoContamination] = ...
          getDefectedImage(this, size)
      [imageNoContamination, isNonNullImage] = this.getDefectedImage@DefectSimulator(size);
      [imageNoContamination, isNonNullImage] = ...
          this.addDust(imageNoContamination, isNonNullImage, this.p, this.altMean, this.altStd);
      imageContaminated = this.multiply(imageNoContamination, this.multiplier);
      imageContaminated = this.addPlane(imageContaminated, this.grad);
    end
    
  end
  
end

