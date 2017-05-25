%ABSTRACT CLASS: PANEL COUNTER
%Instantised objects from this class serve the purpose with return the coordinates of each panel in an iterative fashion
%To use this class:
    %instantise an object from this class
    %use this.hasNextPanelCorner to see if there are more panel coordinates to be obtaiend
    %use this.getNextPanelCorner to get the top-left and bottom right coordinates of this panel
        %once this method is called, the iterator will move to the next panel
        %the iterator should move row by row, then column by column (i.e. downwards)
classdef PanelCounter < handle
    
    %MEMBER VARIABLES
    properties
        panel_height; %height of the panels
        panel_width; %width of the panels
        n_panel_column; %number of columns of panels
        n_panel; %number of panels
        
        i_row; %iterator, counting the number of panel columns
        i_column; %iterator, counting the number of panel rows
    end
    
    %METHODS
    methods
        
        %CONSTRUCTOR
        function this = PanelCounter()
            %get the iterator member variables
            this.resetPanelCorner();
        end
        
        %RESET PANEL CORNER
        %Reset the iteartor for obtaining the corners of the panel
        function resetPanelCorner(this)
            this.i_row = 1;
            this.i_column = 1;
        end
        
        %HAS NEXT PANEL CORNER
        %Output boolean, true if the iterator has another panel to iterate through
        function has_next = hasNextPanelCorner(this)
            %if the iterator has counted beyound the number of columns
            if (this.i_column > this.n_panel_column)
                %the iterator has no more panels to iterator
                has_next = false;
            %else it has, return true
            else
                has_next = true;
            end
        end
        
    end
    
    %ABSTRACT METHODS
    methods (Abstract)
        
        %GET NEXT PANEL CORNER
        %Get the top left and bottom right coordinates of the next panel in
        %the iterator
        %RETURN:
            %corner_position: 2x2 matrix, each column cotains the
            %coordinates of the top left and bottom right of the panel. The
            %coordinates are in the form of matrix index, i.e. 1st row is
            %for the height, 2nd row for the width.
        corner_position = getNextPanelCorner(this);
        
    end
    
end

