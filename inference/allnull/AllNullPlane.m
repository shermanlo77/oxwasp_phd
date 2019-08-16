%MIT License
%Copyright (c) 2019 Sherman Lo

%CLASS: EXPERIMENT ALL NULL PALNE
%See superclass Experiment_AllNull
%Experiment with the empirical null filter on a gaussian image x 2 + plane
classdef AllNullPlane < AllNull
  
  properties (SetAccess = private)
    trueNullMeanGrad = [0.01, 0.01]; %gradient of the plane
    trueNullStd = 2; %variance of the source
    defectSimulator; %object for adding the plane
  end
  
  methods (Access = public)
    
    %CONSTRUCTOR
    function this = AllNullPlane()
      this@AllNull();
    end
    
  end
  
  methods (Access = protected)
    
    %OVERRIDE: SETUP
    function setup(this, seed)
      this.setup@AllNull(seed);
      this.defectSimulator = PlaneMult(this.randStream, this.trueNullMeanGrad, this.trueNullStd);
    end
    
    %IMPLEMENTED: GET IMAGE
    function image = getImage(this)
      image = this.defectSimulator.getDefectedImage(this.imageSize);
    end
    
    %IMPLEMENTED: GET Y LIM
    function yLim = getYLim(this, graphIndex)
      yLim = [];
      switch graphIndex
        case 1
          yLim = [-4, 4];
        case 2
          yLim = [1, 3.5];
        case 3
          yLim = [-0.06, 0.06];
        case 4
          yLim = [0.88, 1.03];
        case 5
          yLim = [2.95, 3.5];
      end
    end
    
  end
  
end

