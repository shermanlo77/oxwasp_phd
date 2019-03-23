%CLASS: EXPERIMENT DEFECT SIMULATION DUST
%Experiment on a 256 x 256 Gaussian which has been defected with dust and then contaminated by a
    %multiplier and a plane.
%Dust defect: all pixels have distribution nullP N(0,1) + altP N(\mu,1), all defects are randomly
    %positioned.
%Various \mu are investigated. For a given \mu, the image is produced with the defect and
    %with/without the contamination. The empirical null filter is used to recover the image after
    %contamination. The 2 images performance on hypothesis testing (picking up the alt pixels or
    %dust) are investigated. For a given FDR level, the type 1, type 2 error and FDR are recorded.
    %By varyin the FDR level, the area of the ROC is obtained. This is repeated multiple times by
    %obtaining a different image
%Plots the following: type 1, type 2, area of ROC and FDR vs alternative mean for images pre/post
    %contamination
classdef DefectAlt < Experiment

  properties (SetAccess = private)
    
    imageSize = 256; %dimension of the image
    radius = 20; %radius of the empirical null filter kernel
    randStream; %rng
    nIntial = 3; %number of initial points used for the empirical null filter
    nRoc = 1000; %number of points used for the roc curve
    altMeanArray = linspace(1,5,9); %array of alt distribution means to investigate
    
    %records results
      %dim 1: for each repeat
      %dim 2: for each alt mean
    type1ErrorArray;
    type2ErrorArray;
    fdrArray;
    rocAreaArray;
    
  end
  
  properties (SetAccess = protected)
    nRepeat = 100; %number of times to repeat the experiment
  end
  
  methods (Access = public)
    
    %CONSTRUCTOR
    function this = DefectAlt()
      this@Experiment();
    end
    
    %IMPLEMENTED: PRINT RESULTS
    %Plots type 1, type 2, area of ROC vs alternative mean
    %Plot in addition the no contamination and contamination ROC area, these are supplied by passing
        %the respective DefectAlt experiments
    %Plot in addition the no contamination quantiles for the errors, again, supplied by passing the
        %respective DefectAlt experiment baseline0
    %How to use:
      %Override with a printResult with no parameters
    function printResults(this, baseline0, baseline)
      
      directory = fullfile('reports','figures','inference','defectsimulation');
      
      %print radius
      fildId = fopen(fullfile(directory,strcat(this.experiment_name,'_radius.txt')),'w');
      fprintf(fildId,'%d',this.radius);
      fclose(fildId);
      
      %print nRepeat
      fildId = fopen(fullfile(directory,strcat(this.experiment_name,'_nRepeat.txt')),'w');
      fprintf(fildId,'%d',this.nRepeat);
      fclose(fildId);
      
      %Plots pre/post contamination results on the same graph, so offset them
      offset = 0.06;
      baseLineConfidence = 0.95;
      baseLineQuantile = (1 - baseLineConfidence)/2;
      
      %plot roc area vs alt mean
      fig = LatexFigure.sub();
      ax = gca;
      
      boxplotFiltered = Boxplots(this.rocAreaArray, true);
      boxplotFiltered.setPosition(this.altMeanArray);
      boxplotFiltered.setColour(ax.ColorOrder(1,:));
      boxplotFiltered.setWantMedian(false);
      boxplotFiltered.setWantTrend(true);
      boxplotFiltered.plot();
      boxplotLegend = boxplotFiltered.getLegendAx();
      label = {'filtered'};
      if (~isempty(baseline0))
        hold on;
        boxplotPreCont = Boxplots(baseline0.rocAreaArray, true);
        boxplotPreCont.setPosition(baseline0.altMeanArray);
        boxplotPreCont.setColour(ax.ColorOrder(2,:));
        boxplotPreCont.setWantMedian(false);
        boxplotPreCont.setWantTrend(true);
        boxplotPreCont.plot();
        boxplotLegend = [boxplotLegend, boxplotPreCont.getLegendAx()];
        label{numel(boxplotLegend)} = 'no contamination';
      end
      if (~isempty(baseline))
        hold on;
        boxplotPostCont = Boxplots(baseline.rocAreaArray, true);
        boxplotPostCont.setPosition(baseline.altMeanArray);
        boxplotPostCont.setColour(ax.ColorOrder(3,:));
        boxplotPostCont.setWantMedian(false);
        boxplotPostCont.setWantTrend(true);
        boxplotPostCont.plot();
        boxplotLegend = [boxplotLegend, boxplotPostCont.getLegendAx()];
        label{numel(boxplotLegend)} = 'contaminated';
      end
      xlabel('alt distribution mean');
      ylabel('ROC area');
      ylim([0.5,1]);
      ax.XLim(1) = this.altMeanArray(1) - offset*2;
      ax.XLim(2) = this.altMeanArray(end) + offset*2;    
      legend(boxplotLegend, label, 'Location', 'southeast');
      saveas(fig,fullfile(directory, strcat(this.experiment_name,'_roc.eps')),'epsc');
      
      %plot type 1 error vs alt mean
      %omit the contaminted plot as this very off the scale compared to the non-contaminted and
          %filtered
      fig = LatexFigure.sub();
      ax = gca;
      boxplotFiltered = Boxplots(this.type1ErrorArray, true);
      boxplotFiltered.setPosition(this.altMeanArray);
      boxplotFiltered.setColour(ax.ColorOrder(1,:));
      boxplotFiltered.plot();
      if (~isempty(baseline0))
        hold on;
        baseType1Error = ...
            quantile(baseline0.type1ErrorArray, [baseLineQuantile, 1-baseLineQuantile]);
        plot(baseline0.altMeanArray, baseType1Error(1,:), 'k--');
        plot(baseline0.altMeanArray, baseType1Error(2,:), 'k--');
      end
      xlabel('alt distribution mean');
      ylabel('type 1 error');
      ylim([0,0.012]);
      ax.XLim(1) = this.altMeanArray(1) - offset*2;
      ax.XLim(2) = this.altMeanArray(end) + offset*2;
      saveas(fig,fullfile(directory, strcat(this.experiment_name,'_type1.eps')),'epsc');
      
      %plot type 2 error vs alt mean
      fig = LatexFigure.sub();
      ax = gca;
      boxplotFiltered = Boxplots(this.type2ErrorArray, true);
      boxplotFiltered.setPosition(this.altMeanArray);
      boxplotFiltered.setColour(ax.ColorOrder(1,:));
      boxplotFiltered.plot();
      if (~isempty(baseline0))
        hold on;
        baseType2Error = ...
            quantile(baseline0.type2ErrorArray, [baseLineQuantile, 1-baseLineQuantile]);
        plot(baseline0.altMeanArray, baseType2Error(1,:), 'k--');
        plot(baseline0.altMeanArray, baseType2Error(2,:), 'k--');
      end
      xlabel('alt distribution mean');
      ylabel('type 2 error');
      ax.XLim(1) = this.altMeanArray(1) - offset*2;
      ax.XLim(2) = this.altMeanArray(end) + offset*2;
      saveas(fig,fullfile(directory, strcat(this.experiment_name,'_type2.eps')),'epsc');
      
      %fdr vs alt mean
      fig = LatexFigure.sub();
      ax = gca;
      boxplotFiltered = Boxplots(this.fdrArray, true);
      boxplotFiltered.setPosition(this.altMeanArray);
      boxplotFiltered.setColour(ax.ColorOrder(1,:));
      boxplotFiltered.plot();
      if (~isempty(baseline0))
        hold on;
        baseFdr = ...
            quantile(baseline0.fdrArray, [baseLineQuantile, 1-baseLineQuantile]);
        plot(baseline0.altMeanArray, baseFdr(1,:), 'k--');
        plot(baseline0.altMeanArray, baseFdr(2,:), 'k--');
      end
      xlabel('alt distribution mean');
      ylabel('fdr');
      ax.XLim(1) = this.altMeanArray(1) - offset*2;
      ax.XLim(2) = this.altMeanArray(end) + offset*2;
      saveas(fig,fullfile(directory, strcat(this.experiment_name,'_fdr.eps')),'epsc');
      
    end
    
  end
  
  methods (Access = protected)
    
    %METHOD: SETUP
    function setup(this, seed)
      this.type1ErrorArray = zeros(this.nRepeat, numel(this.altMeanArray));
      this.type2ErrorArray = zeros(this.nRepeat, numel(this.altMeanArray));
      this.fdrArray = zeros(this.nRepeat, numel(this.altMeanArray));
      this.rocAreaArray = zeros(this.nRepeat, numel(this.altMeanArray));
      this.randStream = RandStream('mt19937ar','Seed', seed);
    end
    
    %METHOD: DO EXPERIMENT
    function doExperiment(this)
      
      %for each alt mean
      for iMu = 1:numel(this.altMeanArray)
        
        %set up the contamination
        defectSimulator = this.getDefectSimulator(this.altMeanArray(iMu));
        
        %repeat nRepeat times
        for iRepeat = 1:this.nRepeat
          
          %get the defected image
          [imageContaminated, isNonNullImage] = ...
              defectSimulator.getDefectedImage([this.imageSize, this.imageSize]);
            
          %filter it
          %get the empirical null and the filtered image
          imageFiltered = this.filterImage(imageContaminated);
          %do the hypothesis testing for the 3 images
          zTester = ZTester(imageFiltered);
          zTester.doTest();

          %get the roc area
          [~, ~, this.rocAreaArray(iRepeat, iMu)] = roc(imageFiltered, isNonNullImage, this.nRoc);
          
          %get the error rates fdrArray
          this.type1ErrorArray(iRepeat, iMu) = ...
            sum(zTester.positiveImage(~isNonNullImage)) / sum(sum(~isNonNullImage));
          
          this.type2ErrorArray(iRepeat, iMu) = ...
              sum(~(zTester.positiveImage(isNonNullImage))) / sum(sum(isNonNullImage));
          
          nSig = sum(sum(zTester.positiveImage));
          if nSig == 0
            fdr = 0;
          else
            fdr = sum(sum(zTester.positiveImage(~isNonNullImage))) / nSig;
          end
          this.fdrArray(iRepeat, iMu) = fdr;

          this.printProgress( ((iMu-1)*this.nRepeat + iRepeat) ... 
              / (numel(this.altMeanArray) * this.nRepeat) );
          
        end
        
      end
      
    end
    
    %METHOD: GET FILETERED IMAGE
    %Given an image, filter it
    function imageFiltered = filterImage(this, imageContaminated)
      filter = this.getFilter();
      filter.setNInitial(this.nIntial);
      filter.filter(imageContaminated);
      imageFiltered = filter.getFilteredImage();
    end
    
  end
  
  methods (Abstract, Access = protected)
    
    %ABSTRACT METHOD: GET DEFECT SIMULATOR
    defectSimulator = getDefectSimulator(this, altMean);
    
    %ABSTRACT METHOD: GET FILTER
    %Return an instantiated a null filter
    filter = getFilter(this)
    
  end
  
end
