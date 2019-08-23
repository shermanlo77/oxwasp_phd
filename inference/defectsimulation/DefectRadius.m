%MIT License
%Copyright (c) 2019 Sherman Lo

%ABSTRACT CLASS: DEFECT RADIUS EXPERIMENT
%Investigate the performance of the empirical null filter with different kernel radius on a defect
    %image pre/post contamination with a plane and a multiplier
%For a given radius, produce defected image pre/post contamination. Use the empirical null filter to
    %recover the image from contamination. These 2 images are then used to do a hypothesis test to
    %find the defects. The type 1 error, type 2 error, fdr and area of roc are recorded. The
    %experiment is repeated by producing another image.
classdef DefectRadius < Experiment

  properties (SetAccess = protected)
    
    nRoc = 1000; %for roc function
    nRepeat = 100; %number of times to repeat the experiment
    imageSize = 256; %dimension of the image
    radiusArray = 10:10:100; %radius of the empirical null filter kernel
    randStream; %rng
    nIntial = 3; %number of initial points used for the empirical null filter
    
    altMean = 3; %mean of the alt distribution
    altStd = 1; %std of the alt distribution
    gradContamination = [0.01, 0.01]; %gradient of the contamination
    multContamination = 2; %multiplier of the contamination
    
    %records results
      %dim 1: for each repeat
      %dim 2: for each radius
      %dim 3: size 2, pre contamination and filtered
    type1ErrorArray;
    type2ErrorArray;
    fdrArray;
    rocAreaArray;
    
  end
  
  methods (Access = public)
    
    %CONSTRUCTOR
    function this = DefectRadius()
      this@Experiment();
    end
    
    %METHOD: PRINT RESULTS
    %Plots type 1, type 2, area of ROC vs alternative mean
    %Plots pre/post contamination results on the same graph
    function printResults(this)
      
      directory = fullfile('reports','figures','inference');
      offset = 10;
      
      %plot roc area vs alt mean
      fig = LatexFigure.sub();
      ax = gca;
      boxplot = Boxplots(this.rocAreaArray(:,:,2), true);
      boxplot.setPosition(this.radiusArray);
      boxplot.plot();
      hold on;
      oracleInterval = quantile(reshape(this.rocAreaArray(:,:,1),[],1), normcdf([-2,2]));
      ax.XLim(1) = min(this.radiusArray) - offset;
      ax.XLim(2) = max(this.radiusArray) + offset;
      plot(ax.XLim, [oracleInterval(1), oracleInterval(1)], 'k--');
      plot(ax.XLim, [oracleInterval(2), oracleInterval(2)], 'k--');
      xlabel('radius (px)');
      ylabel('AUC');
      saveas(fig,fullfile(directory, strcat(this.experimentName,'_roc.eps')),'epsc');
      
      %plot type 1 error vs alt mean
      fig = LatexFigure.sub();
      ax = gca;
      boxplot = Boxplots(this.type1ErrorArray(:,:,2), true);
      boxplot.setPosition(this.radiusArray);
      boxplot.plot();
      hold on;
      oracleInterval = quantile(reshape(this.type1ErrorArray(:,:,1),[],1), normcdf([-2,2]));
      ax.XLim(1) = min(this.radiusArray) - offset;
      ax.XLim(2) = max(this.radiusArray) + offset;
      plot(ax.XLim, [oracleInterval(1), oracleInterval(1)], 'k--');
      plot(ax.XLim, [oracleInterval(2), oracleInterval(2)], 'k--');
      xlabel('radius (px)');
      ylabel('type 1 error');
      saveas(fig,fullfile(directory, strcat(this.experimentName,'_type1.eps')),'epsc');
      
      %plot type 2 error vs alt mean
      fig = LatexFigure.sub();
      ax = gca;
      boxplot = Boxplots(this.type2ErrorArray(:,:,2), true);
      boxplot.setPosition(this.radiusArray);
      boxplot.plot();
      hold on;
      oracleInterval = quantile(reshape(this.type2ErrorArray(:,:,1),[],1), normcdf([-2,2]));
      ax.XLim(1) = min(this.radiusArray) - offset;
      ax.XLim(2) = max(this.radiusArray) + offset;
      plot(ax.XLim, [oracleInterval(1), oracleInterval(1)], 'k--');
      plot(ax.XLim, [oracleInterval(2), oracleInterval(2)], 'k--');
      xlabel('radius (px)');
      ylabel('type 2 error');
      saveas(fig,fullfile(directory, strcat(this.experimentName,'_type2.eps')),'epsc');
      
      %fdr vs alt mean
      fig = LatexFigure.sub();
      ax = gca;
      boxplot = Boxplots(this.fdrArray(:,:,2), true);
      boxplot.setPosition(this.radiusArray);
      boxplot.plot();
      hold on;
      oracleInterval = quantile(reshape(this.fdrArray(:,:,1),[],1), normcdf([-2,2]));
      ax.XLim(1) = min(this.radiusArray) - offset;
      ax.XLim(2) = max(this.radiusArray) + offset;
      plot(ax.XLim, [oracleInterval(1), oracleInterval(1)], 'k--');
      plot(ax.XLim, [oracleInterval(2), oracleInterval(2)], 'k--');
      xlabel('radius (px)');
      ylabel('fdr');
      saveas(fig,fullfile(directory, strcat(this.experimentName,'_fdr.eps')),'epsc');
      
    end
    
  end
  
  methods (Access = protected)
    
    %METHOD: SETUP
    function setup(this, seed)
      this.type1ErrorArray = zeros(this.nRepeat, numel(this.radiusArray), 2);
      this.type2ErrorArray = zeros(this.nRepeat, numel(this.radiusArray), 2);
      this.fdrArray = zeros(this.nRepeat, numel(this.radiusArray), 2);
      this.rocAreaArray = zeros(this.nRepeat, numel(this.radiusArray), 2);
      this.randStream = RandStream('mt19937ar','Seed', seed);
    end
    
    %METHOD: DO EXPERIMENT
    function doExperiment(this)
      
      %set up the contamination
      defectSimulator = this.getDefectSimulator();
      
      %for each kernel radius
      for iRadius = 1:numel(this.radiusArray)
        
        %repeat nRepeat times
        for iRepeat = 1:this.nRepeat
          
          %get the defected image pre/post contamination
          [imagePostCont, isNonNullImage, imagePreCont] = ...
              defectSimulator.getDefectedImage([this.imageSize, this.imageSize]);

          %filter it
          filter = EmpiricalNullFilter(this.radiusArray(iRadius));
          filter.setNInitial(this.nIntial);
          filter.setSeed(this.randStream.randi([intmin('int32'),intmax('int32')],'int32'));
          filter.filter(imagePostCont);

          %get the empirical null and the filtered image
          imageFiltered = filter.getFilteredImage();

          %get the image pre/post bias with significant pixels highlighted
          zTesterPreCont = ZTester(imagePreCont);
          zTesterPreCont.doTest();
          zTesterFiltered = ZTester(imageFiltered);
          zTesterFiltered.doTest();

          %get the roc area
          [~, ~, this.rocAreaArray(iRepeat, iRadius, 1)] = roc(imagePreCont, isNonNullImage, ...
              this.nRoc);
          [~, ~, this.rocAreaArray(iRepeat, iRadius, 2)] = roc(imageFiltered, isNonNullImage, ...
              this.nRoc);
          
          %get the error rates fdrArray
          
          this.type1ErrorArray(iRepeat, iRadius, 1) = ...
              sum(zTesterPreCont.positiveImage(~isNonNullImage)) / sum(sum(~isNonNullImage));
          this.type1ErrorArray(iRepeat, iRadius, 2) = ...
              sum(zTesterFiltered.positiveImage(~isNonNullImage)) / sum(sum(~isNonNullImage));
            
          this.type2ErrorArray(iRepeat, iRadius, 1) = ...
              sum(~(zTesterPreCont.positiveImage(isNonNullImage))) / sum(sum(isNonNullImage));
          this.type2ErrorArray(iRepeat, iRadius, 2) = ...
              sum(~(zTesterFiltered.positiveImage(isNonNullImage))) / sum(sum(isNonNullImage));
            
          nPositive = sum(sum(zTesterPreCont.positiveImage));
          if nPositive == 0
            fdr = 0;
          else
            fdr = sum(sum(zTesterPreCont.positiveImage(~isNonNullImage))) / nPositive;
          end
          this.fdrArray(iRepeat, iRadius, 1) = fdr;
          
          nPositive = sum(sum(zTesterFiltered.positiveImage));
          if nPositive == 0
            fdr = 0;
          else
            fdr = sum(sum(zTesterFiltered.positiveImage(~isNonNullImage))) / nPositive;
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

