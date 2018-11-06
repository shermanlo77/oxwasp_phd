%CLASS: EXPERIMENT ALL NULL PALNE
%See superclass Experiment_AllNull
%Experiment with the empirical null filter on a gaussian image x 2 + plane
classdef Experiment_AllNullPlane < Experiment_AllNull
  
  properties (SetAccess = private)
    trueNullMeanGrad = 0.01; %gradient of the plane
    trueNullStd = 2; %variance of the source
    defectSimulator; %object for adding the plane
  end
  
  methods (Access = public)
    
    %CONSTRUCTOR
    function this = Experiment_AllNullPlane()
      this@Experiment_AllNull('Experiment_AllNullPlane');
    end
    
  end
  
  methods (Access = protected)
    
    %METHOD: SETUP
    function setup(this)
      this.setup@Experiment_AllNull(uint32(2084672537));
      this.defectSimulator = DefectSimulator([this.imageSize(1), this.imageSize(2)]);
      this.defectSimulator.addPlane([this.trueNullMeanGrad, this.trueNullMeanGrad]);
    end
    
    %METHOD: GET IMAGE
    function image = getImage(this)
      %create pure gaussian image and defect it
      image = this.randStream.randn(this.imageSize(1), this.imageSize(2));
      image = image * this.trueNullStd;
      image = this.defectSimulator.defectImage(image);
    end
    
  end
  
end

