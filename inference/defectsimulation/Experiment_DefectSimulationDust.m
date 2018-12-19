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
classdef Experiment_DefectSimulationDust < Experiment

  properties (SetAccess = private)
    
    nRepeat = 100; %number of times to repeat the experiment
    imageSize = 256; %dimension of the image
    radius = 20; %radius of the empirical null filter kernel
    randStream = RandStream('mt19937ar','Seed',uint32(153380491)); %rng
    nIntial = 3; %number of initial points used for the empirical null filter
    nRoc = 1000; %number of points used for the roc curve
    
    altMeanArray = linspace(1,5,9); %array of alt distribution means to investigate
    altStd = 1; %std of the alt distribution
    gradContamination = [0.01, 0.01]; %gradient of the contamination
    multContamination = 2; %multiplier of the contamination
    altP = 0.1; %proportion of pixels which are alt
    
    %records results
      %dim 1: for each repeat
      %dim 2: for each alt mean
      %dim 3: size 2, pre-contamination, contaminated, filtered contaminated
    type1ErrorArray;
    type2ErrorArray;
    fdrArray;
    rocAreaArray;
    
  end
  
  methods (Access = public)
    
    %CONSTRUCTOR
    function this = Experiment_DefectSimulationDust()
      this@Experiment('Experiment_DefectSimulationDust');
    end
    
    %METHOD: PRINT RESULTS
    %Plots type 1, type 2, area of ROC vs alternative mean
    %Plots pre/post contamination and filtered results on the same graph
    function printResults(this)
      
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
      
      %plot roc area vs alt mean
      fig = LatexFigure.sub();
      ax = gca;
      boxplotPreCont = Boxplots(this.rocAreaArray(:,:,1), true);
      boxplotPreCont.setPosition(this.altMeanArray - offset);
      boxplotPreCont.setColour(ax.ColorOrder(1,:));
      boxplotPreCont.plot();
      hold on;
      boxplotPostCont = Boxplots(this.rocAreaArray(:,:,2), true);
      boxplotPostCont.setPosition(this.altMeanArray);
      boxplotPostCont.setColour(ax.ColorOrder(2,:));
      boxplotPostCont.plot();
      boxplotFiltered = Boxplots(this.rocAreaArray(:,:,3), true);
      boxplotFiltered.setPosition(this.altMeanArray + offset);
      boxplotFiltered.setColour(ax.ColorOrder(3,:));
      boxplotFiltered.plot();
      xlabel('alt distribution mean');
      ylabel('ROC area');
      ax.XLim(1) = this.altMeanArray(1) - offset*2;
      ax.XLim(2) = this.altMeanArray(end) + offset*2;
      boxplotLegend = [boxplotPreCont.getLegendAx(), boxplotPostCont.getLegendAx(), ...
          boxplotFiltered.getLegendAx()];
      legend(boxplotLegend, 'pre contamination', 'contaminated', 'filtered', ...
          'Location', 'southeast');
      saveas(fig,fullfile(directory, strcat(this.experiment_name,'_roc.eps')),'epsc');
      
      %plot type 1 error vs alt mean
      %omit the contaminted plot as this very off the scale compared to the non-contaminted and
          %filtered
      fig = LatexFigure.sub();
      ax = gca;
      boxplotPreCont = Boxplots(this.type1ErrorArray(:,:,1), true);
      boxplotPreCont.setPosition(this.altMeanArray - offset);
      boxplotPreCont.setColour(ax.ColorOrder(1,:));
      boxplotPreCont.plot();
      hold on;
      boxplotFiltered = Boxplots(this.type1ErrorArray(:,:,3), true);
      boxplotFiltered.setPosition(this.altMeanArray + offset);
      boxplotFiltered.setColour(ax.ColorOrder(3,:));
      boxplotFiltered.plot();
      xlabel('alt distribution mean');
      ylabel('type 1 error');
      ax.XLim(1) = this.altMeanArray(1) - offset*2;
      ax.XLim(2) = this.altMeanArray(end) + offset*2;
      boxplotLegend = [boxplotPreCont.getLegendAx(), boxplotFiltered.getLegendAx()];
      legend(boxplotLegend, 'pre contamination', 'filtered', 'Location', 'northwest');
      saveas(fig,fullfile(directory, strcat(this.experiment_name,'_type1.eps')),'epsc');
      
      %plot type 2 error vs alt mean
      fig = LatexFigure.sub();
      ax = gca;
      boxplotPreCont = Boxplots(this.type2ErrorArray(:,:,1), true);
      boxplotPreCont.setPosition(this.altMeanArray - offset);
      boxplotPreCont.setColour(ax.ColorOrder(1,:));
      boxplotPreCont.plot();
      hold on;
      boxplotPostCont = Boxplots(this.type2ErrorArray(:,:,2), true);
      boxplotPostCont.setPosition(this.altMeanArray);
      boxplotPostCont.setColour(ax.ColorOrder(2,:));
      boxplotPostCont.plot();
      boxplotFiltered = Boxplots(this.type2ErrorArray(:,:,3), true);
      boxplotFiltered.setPosition(this.altMeanArray + offset);
      boxplotFiltered.setColour(ax.ColorOrder(3,:));
      boxplotFiltered.plot();
      xlabel('alt distribution mean');
      ylabel('type 2 error');
      ax.XLim(1) = this.altMeanArray(1) - offset*2;
      ax.XLim(2) = this.altMeanArray(end) + offset*2;
      boxplotLegend = [boxplotPreCont.getLegendAx(), boxplotPostCont.getLegendAx(), ...
          boxplotFiltered.getLegendAx()];
      legend(boxplotLegend, 'pre contamination', 'contaminated', 'filtered', ...
          'Location', 'northeast');
      saveas(fig,fullfile(directory, strcat(this.experiment_name,'_type2.eps')),'epsc');
      
      %fdr vs alt mean
      fig = LatexFigure.sub();
      ax = gca;
      boxplotPreCont = Boxplots(this.fdrArray(:,:,1), true);
      boxplotPreCont.setPosition(this.altMeanArray - offset);
      boxplotPreCont.setColour(ax.ColorOrder(1,:));
      boxplotPreCont.plot();
      hold on;
      boxplotPostCont = Boxplots(this.fdrArray(:,:,2), true);
      boxplotPostCont.setPosition(this.altMeanArray);
      boxplotPostCont.setColour(ax.ColorOrder(2,:));
      boxplotPostCont.plot();
      boxplotFiltered = Boxplots(this.type2ErrorArray(:,:,3), true);
      boxplotFiltered.setPosition(this.altMeanArray + offset);
      boxplotFiltered.setColour(ax.ColorOrder(3,:));
      boxplotFiltered.plot();
      xlabel('alt distribution mean');
      ylabel('fdr');
      ax.XLim(1) = this.altMeanArray(1) - offset*2;
      ax.XLim(2) = this.altMeanArray(end) + offset*2;
      boxplotLegend = [boxplotPreCont.getLegendAx(), boxplotPostCont.getLegendAx(), ...
          boxplotFiltered.getLegendAx()];
      legend(boxplotLegend, 'pre contamination', 'contaminated', 'filtered', ...
          'Location', 'best');
      saveas(fig,fullfile(directory, strcat(this.experiment_name,'_fdr.eps')),'epsc');
      
    end
    
  end
  
  methods (Access = protected)
    
    %METHOD: SETUP
    function setup(this)
      this.type1ErrorArray = zeros(this.nRepeat, numel(this.altMeanArray), 3);
      this.type2ErrorArray = zeros(this.nRepeat, numel(this.altMeanArray), 3);
      this.fdrArray = zeros(this.nRepeat, numel(this.altMeanArray), 3);
      this.rocAreaArray = zeros(this.nRepeat, numel(this.altMeanArray), 3);
    end
    
    %METHOD: DO EXPERIMENT
    function doExperiment(this)
      
      %for each alt mean
      for iMu = 1:numel(this.altMeanArray)
        
        %set up the contamination
        defectSimulator = PlaneMultDust(this.randStream, this.gradContamination, ...
            this.multContamination, this.altP, this.altMeanArray(iMu), this.altStd);
        
        %repeat nRepeat times
        for iRepeat = 1:this.nRepeat
          
          %get the defected image
          [imageContaminated, isAltImage, imagePreContaminated] = ...
              defectSimulator.getDefectedImage([this.imageSize, this.imageSize]);
            
          %filter it
          filter = EmpiricalNullFilter(this.radius);
          filter.setNInitial(this.nIntial);
          filter.filter(imageContaminated);

          %get the empirical null and the filtered image
          imageFiltered = filter.getFilteredImage();

          %do the hypothesis testing for the 3 images
          zTesterPreContaminated = ZTester(imagePreContaminated);
          zTesterPreContaminated.doTest();
          zTesterContaminated = ZTester(imageContaminated);
          zTesterContaminated.doTest();
          zTesterFiltered = ZTester(imageFiltered);
          zTesterFiltered.doTest();

          %get the roc area
          [~, ~, this.rocAreaArray(iRepeat, iMu, 1)] = ...
              roc(imagePreContaminated, isAltImage, this.nRoc);
          [~, ~, this.rocAreaArray(iRepeat, iMu, 2)] = ...
              roc(imageContaminated, isAltImage, this.nRoc);
          [~, ~, this.rocAreaArray(iRepeat, iMu, 3)] = ...
              roc(imageFiltered, isAltImage, this.nRoc);
          
          %get the error rates fdrArray
          
          this.type1ErrorArray(iRepeat, iMu, 1) = ...
              sum(zTesterPreContaminated.sig_image(~isAltImage)) / sum(sum(~isAltImage));
          this.type1ErrorArray(iRepeat, iMu, 2) = ...
              sum(zTesterContaminated.sig_image(~isAltImage)) / sum(sum(~isAltImage));
          this.type1ErrorArray(iRepeat, iMu, 3) = ...
            sum(zTesterFiltered.sig_image(~isAltImage)) / sum(sum(~isAltImage));
            
          this.type2ErrorArray(iRepeat, iMu, 1) = ...
              sum(~(zTesterPreContaminated.sig_image(isAltImage))) / sum(sum(isAltImage));
          this.type2ErrorArray(iRepeat, iMu, 2) = ...
              sum(~(zTesterContaminated.sig_image(isAltImage))) / sum(sum(isAltImage));
          this.type2ErrorArray(iRepeat, iMu, 3) = ...
              sum(~(zTesterFiltered.sig_image(isAltImage))) / sum(sum(isAltImage));
            
          nSig = sum(sum(zTesterPreContaminated.sig_image));
          if nSig == 0
            fdr = 0;
          else
            fdr = sum(sum(zTesterPreContaminated.sig_image(~isAltImage))) / nSig;
          end
          this.fdrArray(iRepeat, iMu, 1) = fdr;
          
          nSig = sum(sum(zTesterContaminated.sig_image));
          if nSig == 0
            fdr = 0;
          else
            fdr = sum(sum(zTesterContaminated.sig_image(~isAltImage))) / nSig;
          end
          this.fdrArray(iRepeat, iMu, 2) = fdr;
          
          nSig = sum(sum(zTesterFiltered.sig_image));
          if nSig == 0
            fdr = 0;
          else
            fdr = sum(sum(zTesterFiltered.sig_image(~isAltImage))) / nSig;
          end
          this.fdrArray(iRepeat, iMu, 3) = fdr;

          this.printProgress( ((iMu-1)*this.nRepeat + iRepeat) ... 
              / (numel(this.altMeanArray) * this.nRepeat) );
          
        end
        
      end
      
    end
    
  end
  
end

