experiments = {...
  VarMeanCvAbsNoFilterNull();
  VarMeanCvAbsNoFilterBw();
  VarMeanCvAbsNoFilterLinear();
  VarMeanCvAbsFilterNull();
  VarMeanCvAbsFilterBw();
  VarMeanCvAbsFilterLinear();
};

for i = 1:numel(experiments)
  experiments{i}.run();
end

%print results
for iData = 1:2
  
  %get array of the same data but with different shading corrections
  switch iData
    case 1
      experiments = {...
        VarMeanCvAbsNoFilterNull();
        VarMeanCvAbsNoFilterBw();
        VarMeanCvAbsNoFilterLinear();
      };
    case 2
      experiments = {...
        VarMeanCvAbsFilterNull();
        VarMeanCvAbsFilterBw();
        VarMeanCvAbsFilterLinear();
      };
  end
  
  %use just an experiment to get properties of the experiment
  this = experiments{1};
  
  %get list of shading correction
  shadingCorrectionNameArray = cell(size(experiments));
  for iShadingCorrection = 1:numel(shadingCorrectionNameArray)
    shadingCorrector = experiments{iShadingCorrection}.scan.shadingCorrector;
    if (isempty(shadingCorrector))
      shadingCorrectionNameArray{iShadingCorrection} = 'null';
    else
      shadingCorrectionNameArray{iShadingCorrection} = shadingCorrector.getName();
    end
  end
  
  %plot the training and test deviance
  for iStat = 1:2
    isTrainingDeviance = iStat == 1;
    if (isTrainingDeviance)
      statName = 'training deviance';
    else
      statName = 'test deviance';
    end
    
    %plot the test deviance for each shading correction
    fig = LatexFigure.sub();
    legendArray = line(0,0);
    for iShadingCorrection = 1:numel(experiments)
      boxplots = experiments{iShadingCorrection}. ...
          printResults(-0.1 + ((iShadingCorrection-1)*0.1), isTrainingDeviance);
      legendArray(iShadingCorrection) = boxplots.getLegendAx();
    end

    %retick the x axis
    ax = fig.Children;
    ax.XTick = 1:numel(this.modelArray);
    %label each glm with its name
    ax.XTickLabel = this.modelArray;
    ax.XLim = [0.5,numel(this.modelArray)+0.5];
    %label the axis and legend
    ylabel(statName);
    legend(legendArray,shadingCorrectionNameArray,'Location','best');
    
    %save the figure
    fileName = fullfile('reports','figures','varmean', ...
        strcat(mfilename,'_',class(this.scan),'_',statName,'.eps'));
    saveas(fig,fileName,'epsc');
  end
end