%ABSTRACT CLASS: DEFECT RADIUS EXPERIMENT
%Investigate the performance of the empirical null filter with different kernel radius on a defect
    %image pre/post contamination with a plane and a multiplier
%For a given radius, produce defected image pre/post contamination. Use the empirical null filter to
    %recover the image from contamination. These 2 images are then used to do a hypothesis test to
    %find the defects. The type 1 error, type 2 error, fdr and area of roc are recorded. The
    %experiment is repeated by producing another image.
classdef Experiment_DefectRadius < Experiment

  properties (SetAccess = protected)
    
    nRepeat = 100; %number of times to repeat the experiment
    imageSize = 256; %dimension of the image
    radiusArray = 10:10:100; %radius of the empirical null filter kernel
    randStream; %rng
    nIntial = 3; %number of initial points used for the empirical null filter
    
    altMean = 2; %mean of the alt distribution
    altStd = 1; %std of the alt distribution
    gradContamination = [0.01, 0.01]; %gradient of the contamination
    multContamination = 2; %multiplier of the contamination
    
    %records results
      %dim 1: for each repeat
      %dim 2: for each radius
      %dim 3: size 2, pre/post contamination
    type1ErrorArray;
    type2ErrorArray;
    fdrArray;
    rocAreaArray;
    
  end
  
  methods (Access = public)
    
    %CONSTRUCTOR
    function this = Experiment_DefectRadius(experimentName)
      this@Experiment(experimentName);
    end
    
    %METHOD: PRINT RESULTS
    %Plots type 1, type 2, area of ROC vs alternative mean
    %Plots pre/post contamination results on the same graph
    function printResults(this)
      
      %Plots pre/post contamination results on the same graph, so offset them
      offset = 0.05;
      
      %plot roc area vs alt mean
      figure;
      ax = gca;
      boxplotPreCont = Boxplots(this.rocAreaArray(:,:,1), true);
      boxplotPreCont.setPosition(this.radiusArray - offset);
      boxplotPreCont.setColour(ax.ColorOrder(1,:));
      boxplotPreCont.plot();
      hold on;
      boxplotPostCont = Boxplots(this.rocAreaArray(:,:,2), true);
      boxplotPostCont.setPosition(this.radiusArray + offset);
      boxplotPostCont.setColour(ax.ColorOrder(2,:));
      boxplotPostCont.plot();
      xlabel('kernel radius');
      ylabel('roc area');
      ax.XLim(1) = this.radiusArray(1) - offset*2;
      ax.XLim(2) = this.radiusArray(end) + offset*2;
      boxplotLegend = [boxplotPreCont.getLegendAx(), boxplotPostCont.getLegendAx()];
      legend(boxplotLegend, 'pre contamination', 'post contamination', 'Location', 'southeast');
      
      %plot type 1 error vs alt mean
      figure;
      ax = gca;
      boxplotPreCont = Boxplots(this.type1ErrorArray(:,:,1), true);
      boxplotPreCont.setPosition(this.radiusArray - offset);
      boxplotPreCont.setColour(ax.ColorOrder(1,:));
      boxplotPreCont.plot();
      hold on;
      boxplotPostCont = Boxplots(this.type1ErrorArray(:,:,2), true);
      boxplotPostCont.setPosition(this.radiusArray + offset);
      boxplotPostCont.setColour(ax.ColorOrder(2,:));
      boxplotPostCont.plot();
      xlabel('kernel radius');
      ylabel('type 1 error');
      ax.XLim(1) = this.radiusArray(1) - offset*2;
      ax.XLim(2) = this.radiusArray(end) + offset*2;
      boxplotLegend = [boxplotPreCont.getLegendAx(), boxplotPostCont.getLegendAx()];
      legend(boxplotLegend, 'pre contamination', 'post contamination', 'Location', 'northwest');
      
      %plot type 2 error vs alt mean
      figure;
      ax = gca;
      boxplotPreCont = Boxplots(this.type2ErrorArray(:,:,1), true);
      boxplotPreCont.setPosition(this.radiusArray - offset);
      boxplotPreCont.setColour(ax.ColorOrder(1,:));
      boxplotPreCont.plot();
      hold on;
      boxplotPostCont = Boxplots(this.type2ErrorArray(:,:,2), true);
      boxplotPostCont.setPosition(this.radiusArray + offset);
      boxplotPostCont.setColour(ax.ColorOrder(2,:));
      boxplotPostCont.plot();
      xlabel('kernel radius');
      ylabel('type 2 error');
      ax.XLim(1) = this.radiusArray(1) - offset*2;
      ax.XLim(2) = this.radiusArray(end) + offset*2;
      boxplotLegend = [boxplotPreCont.getLegendAx(), boxplotPostCont.getLegendAx()];
      legend(boxplotLegend, 'pre contamination', 'post contamination', 'Location', 'southwest');
      
      %fdr vs alt mean
      figure;
      ax = gca;
      boxplotPreCont = Boxplots(this.fdrArray(:,:,1), true);
      boxplotPreCont.setPosition(this.radiusArray - offset);
      boxplotPreCont.setColour(ax.ColorOrder(1,:));
      boxplotPreCont.plot();
      hold on;
      boxplotPostCont = Boxplots(this.fdrArray(:,:,2), true);
      boxplotPostCont.setPosition(this.radiusArray + offset);
      boxplotPostCont.setColour(ax.ColorOrder(2,:));
      boxplotPostCont.plot();
      xlabel('kernel radius');
      ylabel('fdr');
      ax.XLim(1) = this.radiusArray(1) - offset*2;
      ax.XLim(2) = this.radiusArray(end) + offset*2;
      boxplotLegend = [boxplotPreCont.getLegendAx(), boxplotPostCont.getLegendAx()];
      legend(boxplotLegend, 'pre contamination', 'post contamination', 'Location', 'southwest');
      
    end
    
  end
  
  methods (Access = protected)
    
    %METHOD: SETUP
    function setup(this, seed)
      this.type1ErrorArray = zeros(this.nRepeat, numel(this.radiusArray));
      this.type2ErrorArray = zeros(this.nRepeat, numel(this.radiusArray));
      this.fdrArray = zeros(this.nRepeat, numel(this.radiusArray));
      this.rocAreaArray = zeros(this.nRepeat, numel(this.radiusArray));
      this.randStream = RandStream('mt19937ar','Seed', seed);
    end
    
    %METHOD: DO EXPERIMENT
    function doExperiment(this)
      
      %for each alt mean
      for iRadius = 1:numel(this.radiusArray)
        
        %get up the contamination
        defectSimulator = this.getDefectSimulator();
        
        %repeat nRepeat times
        for iRepeat = 1:this.nRepeat
          
          %get the defected image pre/post contamination
          [imagePostCont, isAltImage, imagePreCont] = ...
              defectSimulator.getDefectedImage([this.imageSize, this.imageSize]);

          %filter it
          filter = EmpiricalNullFilter(this.radiusArray(iRadius));
          filter.setNInitial(this.nIntial);
          filter.filter(imagePostCont);

          %get the empirical null and the filtered image
          imageFiltered = filter.getFilteredImage();

          %get the image pre/post bias with significant pixels highlighted
          zTesterPreCont = ZTester(imagePreCont);
          zTesterPreCont.doTest();
          zTesterPostCont = ZTester(imageFiltered);
          zTesterPostCont.doTest();

          %get the roc area
          [~, ~, this.rocAreaArray(iRepeat, iRadius, 1)] = roc(imagePreCont, isAltImage, 100);
          [~, ~, this.rocAreaArray(iRepeat, iRadius, 2)] = roc(imageFiltered, isAltImage, 100);
          
          %get the error rates fdrArray
          
          this.type1ErrorArray(iRepeat, iRadius, 1) = ...
              sum(zTesterPreCont.sig_image(~isAltImage)) / sum(sum(~isAltImage));
          this.type1ErrorArray(iRepeat, iRadius, 2) = ...
              sum(zTesterPostCont.sig_image(~isAltImage)) / sum(sum(~isAltImage));
            
          this.type2ErrorArray(iRepeat, iRadius, 1) = ...
              sum(~(zTesterPreCont.sig_image(isAltImage))) / sum(sum(isAltImage));
          this.type2ErrorArray(iRepeat, iRadius, 2) = ...
              sum(~(zTesterPostCont.sig_image(isAltImage))) / sum(sum(isAltImage));
            
          nSig = sum(sum(zTesterPreCont.sig_image));
          if nSig == 0
            fdr = 0;
          else
            fdr = sum(sum(zTesterPreCont.sig_image(~isAltImage))) / nSig;
          end
          this.fdrArray(iRepeat, iRadius, 1) = fdr;
          
          nSig = sum(sum(zTesterPostCont.sig_image));
          if nSig == 0
            fdr = 0;
          else
            fdr = sum(sum(zTesterPostCont.sig_image(~isAltImage))) / nSig;
          end
          this.fdrArray(iRepeat, iRadius, 2) = fdr;

          this.printProgress( ((iRadius-1)*this.nRepeat + iRepeat) ... 
              / (numel(this.radiusArray) * this.nRepeat) );
          
        end
        
      end
      
    end
    
  end
  
  methods (Abstract, Access = protected)
    
    %ABSTRACT: GET DEFECT SIMULATOR
    %Returns a defect simulator to investigate
    defectSimulator = getDefectSimulator(this);
    
  end
  
end

