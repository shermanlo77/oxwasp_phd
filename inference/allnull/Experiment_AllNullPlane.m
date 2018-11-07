%CLASS: EXPERIMENT ALL NULL PALNE
%See superclass Experiment_AllNull
%Experiment with the empirical null filter on a gaussian image x 2 + plane
classdef Experiment_AllNullPlane < Experiment_AllNull
  
  properties (SetAccess = private)
    trueNullMeanGrad = [0.01, 0.01]; %gradient of the plane
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
      this.defectSimulator = PlaneMult(this.randStream, this.trueNullMeanGrad, this.trueNullStd);
    end
    
    %METHOD: GET IMAGE
    function image = getImage(this)
      image = this.defectSimulator.getDefectedImage(this.imageSize);
    end
    
  end
  
end

