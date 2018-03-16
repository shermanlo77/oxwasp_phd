classdef LatexFigure
    
    properties
    end
    
    methods (Static)
        
        function fig = sub(fig)
            if nargin == 0
                fig = figure('Visible','off');
            else
                fig.Visible = 'off';
            end
            fig.Position(3:4) = [420,315];
        end

        function fig = main(fig)
            if nargin == 0
                fig = figure('Visible','off');
            else
                fig.Visible = 'off';
            end
        end
        
    end
    
end

