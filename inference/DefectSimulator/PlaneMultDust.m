%CLASS: PLANE MULTIPLY
%Produces images which are Gaussian N(0,1) but then multiplied by a scalar followed by an addition
%of a plane
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
      %isAltImage: boolean map, true if that pixel is a defect
      %imagePreBias: defected Gaussian image without the smooth function added
    function [image, isAltImage, imagePreBias] = getDefectedImage(this, size)
      [image, isAltImage] = this.getDefectedImage@DefectSimulator(size);
      [image, isAltImage] = this.addDust(image, isAltImage, this.p, this.altMean, this.altStd);
      imagePreBias = image;
      image = this.multiply(image, this.multiplier);
      image = this.addPlane(image, this.grad);
    end
    
  end
  
end

