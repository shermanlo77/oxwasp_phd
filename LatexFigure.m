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
    
    function fig = main()
      fig = figure('Visible','off');
    end
    
    function convertToPoster(fig)
      fig.Position(3:4) = [2244,1683];
      fig.CurrentAxes.FontSize = 50;
    end
    
  end
  
end

