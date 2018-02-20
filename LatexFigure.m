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
        
    end
    
end

