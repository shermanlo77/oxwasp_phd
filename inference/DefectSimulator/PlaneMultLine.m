%CLASS: PLANE MULTIPLY LINE
%Produces images which are Gaussian N(0,1) but a verticle line will be N(mu, sigma^2) where the alt
    %distribution is dvia the member variables altMean and altStd. This is then multiplied by a
    %scalar followed by an addition of a plane.
classdef PlaneMultLine < PlaneMult
  
  properties (SetAccess = protected)
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
      %altMean: alt distribution mean
      %altStd: alt distribution std
    function this = PlaneMultLine(randStream, grad, multiplier, altMean, altStd)
      this@PlaneMult(randStream, grad, multiplier);
      this.altMean = altMean;
      this.altStd = altStd;
    end
    
    %METHOD: GET DEFECTED IMAGE
    %PARAMETER:
      %size: 2 row vector [height, width]
    %RETURN:
      %image: a defected image
      %isAltImage: boolean map, true if that pixel is a defect
      %imagePreBias: defected Gaussian image without the smooth function added
    function [image, isAltImage, imagePreBias] = getDefectedImage(this, size)
      [image, isAltImage] = this.getDefectedImage@DefectSimulator(size);
      [image, isAltImage] = this.addLineDefect(image, isAltImage, round(size(2)/2), 1, ...
          this.altMean, this.altStd);
      imagePreBias = image;
      image = this.multiply(image, this.multiplier);
      image = this.addPlane(image, this.grad);
    end
    
  end
  
end

