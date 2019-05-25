classdef (Abstract) ShadingCorrectionAnova < Experiment
  
  properties (SetAccess = public)
    
    scan; %scan object to get the calibration images from
    nRepeat = 100; %number of times to repeat the experiment
    rng; %random number generator
    nShadingCorrection = 3; %number of shading corrections used
    shadingCorrectionNameArray; %array of shading correction names
    
    %array of variances of F statistic
      %dim 1: for each repeat
      %dim 2: for each current or power
      %dim 3: for each shading correction
    varBetweenArray;
    varWithinArray;
    
  end
  
  methods (Access = public)
    
    %CONSTRUCTOR
    function this = ShadingCorrectionAnova()
      this@Experiment();
    end
    
    %IMPLEMENTED: PRINT RESULTS
    function printResults(this)
      
      powerArray = this.scan.getPowerArray();
      minVar = min([min(reshape(this.varWithinArray,[],1)), ...
          min(reshape(this.varBetweenArray,[],1))]);
      maxVar = max([max(reshape(this.varWithinArray,[],1)), ...
          max(reshape(this.varBetweenArray,[],1))]);
      
      %for each shading correction, plot the within and between variance
      for iShading = 1:this.nShadingCorrection
        
        fig = LatexFigure.sub();
        ax = gca;
        boxplotWithin = Boxplots(this.varWithinArray(:,:,iShading));
        boxplotWithin.setPosition(powerArray);
        boxplotWithin.plot();
        hold on;
        boxplotBetween = Boxplots(this.varBetweenArray(:,:,iShading));
        boxplotBetween.setPosition(powerArray);
        boxplotBetween.setColour(ax.ColorOrder(2,:));
        boxplotBetween.plot();
        ax.YScale = 'log';
        xlabel('power (W)');
        ylabel('variance');
        legend([boxplotWithin.getLegendAx(), boxplotBetween.getLegendAx()], ...
            'within pixel','between pixel', 'Location', 'best');
        ylim([minVar,maxVar]);
        
      end
      
      %plot the F statistics
      fArray = this.varBetweenArray ./ this.varWithinArray;
      legendAxes(this.nShadingCorrection) = handle(0); %array of axes for lengend
      fig = LatexFigure.sub();
      ax = gca;
      for iShading = 1:this.nShadingCorrection
        boxPlotF = Boxplots(fArray(:,:,iShading));
        boxPlotF.setPosition(powerArray);
        boxPlotF.setColour(ax.ColorOrder(iShading,:));
        boxPlotF.plot();
        legendAxes(iShading) = boxPlotF.getLegendAx();
      end
      ax.YScale = 'log';
      xlabel('power (W)');
      ylabel('F statistic');
      legend(legendAxes, this.shadingCorrectionNameArray);
      
    end
    
  end
  
  methods (Access = protected)
    
    %IMPLEMENTED: SCAN
    %PARAMETERS:
      %scan: scan object containing a valid member variable calibrationArray
      %seed: uint32 for seeding the rng
    function setup(this, scan, seed)
      this.rng = RandStream('mt19937ar', 'Seed', seed);
      this.scan = scan;
      this.varBetweenArray = zeros(this.nRepeat, this.scan.whiteIndex, this.nShadingCorrection);
      this.varWithinArray = zeros(this.nRepeat, this.scan.whiteIndex, this.nShadingCorrection);
      this.shadingCorrectionNameArray = cell(this.nShadingCorrection, 1);
    end
    
    %IMPLEMENTED: DO EXPERIMENT
    function doExperiment(this)
      
      %get the number of replications of the calibration images
      nCalibration = this.scan.calibrationScanArray(1).nSample;
      %one image from each power is held out, the rest is used for variance estimation, there are
          %nTest of them
      nTest = nCalibration - 1;
      nPower = this.scan.whiteIndex; %number of powers used in the calibration
      area = this.scan.area; %area of image
      
      %for each shading correction
      for iShading = 1:this.nShadingCorrection
        
        %nRepeat times
        for iRepeat = 1:this.nRepeat
          
          %imageIndex is a matrix of integers, each column is a randperm
            %dim 1: contains randperm
            %dim 2: for each power
          imageIndex = zeros(nCalibration, nPower);
          for iPower = 1:nPower
            imageIndex(:, iPower) = this.rng.randperm(nCalibration)';
          end
          
          %set shading correction
            %1: no shading correction
            %2: bw shading correction
            %3: shading correction
          switch iShading
            case 2
              this.scan.addShadingCorrectorBw([imageIndex(1,1), imageIndex(1,end)]);
            case 3
              this.scan.addShadingCorrectorLinear(1:nPower, imageIndex(1,:));
          end
          
          %for each power, get the within and between pixel variance
          for iPower = 1:nPower
            %get the shading corrected calibration images
            calibrationImageArray = ...
                this.scan.calibrationScanArray(iPower).loadImageStack(imageIndex(2:end,iPower));
            
            %get the mean for each pixel
            withinPixelMean = mean(calibrationImageArray, 3);
            %get the global mean
            globalMean = mean(reshape(withinPixelMean,[],1));

            %estimate the within pixel variance
            this.varWithinArray(iRepeat, iPower, iShading) = ...
                sum(sum(sum( ( calibrationImageArray - repmat(withinPixelMean,1,1,nTest) ).^2 )))...
                / (area*nTest - area);
            %estimate the between pixel variance
            this.varBetweenArray(iRepeat, iPower, iShading) = ...
                nTest * sum(sum((withinPixelMean - globalMean).^2))/(area-1);
          end
          
          %print progress bar
          this.printProgress( (iRepeat + ((iShading-1)*this.nRepeat)) ...
              / (this.nShadingCorrection*this.nRepeat) );
          
        end
        
        %get the name of the shading correction
        this.shadingCorrectionNameArray{iShading} = this.scan.getShadingCorrectionStatus();
        
      end
      
    end
    
  end
  
end
