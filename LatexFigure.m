classdef LatexFigure
    
    properties
    end
    
    methods (Static)
        
        function fig = sub()
            fig = figure('Visible','on');
            fig.Position(3:4) = [420,315];
        end

        function fig = main()
            fig = figure('Visible','on');
        end
        
    end
    
end

