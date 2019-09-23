%MIT License
%Copyright (c) 2019 Sherman Lo

%CLASS: LATEX FIGURE
%Contain functions for creating a figure with a consistent size
classdef LatexFigure
  
  properties
  end
  
  methods (Static)
    
    function fig = sub()
      fig = figure('Visible','off');
      fig.Position(3:4) = [420,315];
    end
    
    function fig = subLoose()
      fig = figure('Visible','off');
      fig.Position(3:4) = [392,294];
    end
    
    function fig = main()
      fig = figure('Visible','off');
    end
    
    function convertToPoster(fig)
      fig.Position(3:4) = [1400,1600];
      fig.CurrentAxes.FontSize = 35;
    end
    
  end
  
end

