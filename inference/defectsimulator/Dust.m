%CLASS: PLANE MULTIPLY DUST
%Produces images which are Gaussian (1-p)N(0,1) + pN(mu, sigma^2) where the alt distribution is 
    %described via the member variables altMean and altStd. This is then multiplied by a scalar
    %followed by an addition of a plane.
classdef Dust < DefectSimulator
  
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
      %p: probability pixel is alt
      %altMean: alt distribution mean
      %altStd: alt distribution std
    function this = Dust(randStream, p, altMean, altStd)
      this@DefectSimulator(randStream);
      this.isContaminated = false;
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
    function [image, isAltImage] = getDefectedImage(this, size)
      [image, isAltImage] = this.getDefectedImage@DefectSimulator(size);
      [image, isAltImage] = this.addDust(image, isAltImage, this.p, this.altMean, this.altStd);
    end
    
  end
  
end

