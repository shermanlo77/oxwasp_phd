%MIT License
%Copyright (c) 2019 Sherman Lo

%CLASS: PLANE MULTIPLY
%Produces images which are Gaussian N(0,1) but then multiplied by a scalar followed by an addition
%of a plane
classdef PlaneMult < DefectSimulator
  
  properties (SetAccess = protected)
    grad; %2 row vector, gradient of the plane
    multiplier; %scale all pixels by this
  end
  
  methods (Access = public)
    
    %CONSTRUCTOR
    %PARAMETERS:
      %randStream: rng
      %grad: 2 row vector, gradient of the plane
      %multiplier: scale all pixels by this
    function this = PlaneMult(randStream, grad, multiplier)
      this@DefectSimulator(randStream);
      this.isContaminated = true;
      this.grad = grad;
      this.multiplier = multiplier;
    end
    
    %METHOD: GET DEFECTED IMAGE
    %Return an image with all N(0,1) pixels except for the alt pixels x multiplier + plane
    %PARAMETER:
      %size: 2 row vector [height, width]
    %RETURN:
      %image: a defected Gaussian image
      %isNonNullImage: boolean map, true if that pixel is a defect
      %imageNoContamination: image with defect but no contamination
    function [imageContaminated, isNonNullImage, imageNoContamination] = ...
          getDefectedImage(this, size)
      [imageNoContamination, isNonNullImage] = this.getDefectedImage@DefectSimulator(size);
      imageContaminated = this.multiply(imageNoContamination, this.multiplier);
      imageContaminated = this.addPlane(imageContaminated, this.grad);
    end
    
  end
  
end

