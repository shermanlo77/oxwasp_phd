%PANEL COUNTER (BRASS SUBCLASS)
%Brass is just a random noun
%See superclass PanelCounter for derived member variables and methods
classdef PanelCounter_Brass < PanelCounter
    
    %MEMBER VARIABLES
    properties
    end
    
    %METHODS
    methods
        
        %CONSTRUCTOR
        function this = PanelCounter_Brass()
            this@PanelCounter(); %call superclass
            %assign derived member variables
            this.panel_height = 998;
            this.panel_width = 128;
            this.n_panel_column = 16;
            this.n_panel = 32;
        end
        
        %GET NEXT PANEL CORNER
        %Get the top left and bottom right coordinates of the next panel in
        %the iterator
        %RETURN:
            %corner_position: 2x2 matrix, each column cotains the
            %coordinates of the top left and bottom right of the panel. The
            %coordinates are in the form of matrix index, i.e. 1st row is
            %for the height, 2nd row for the width.
        function corner_position = getNextPanelCorner(this)
               
            %declare 2x2 matrix
            corner_position = zeros(2,2);
            
            %get the height_range of the panel
            if this.i_row == 1
                corner_position(1,1) = 1 + (this.i_row-1)*this.panel_height;
                corner_position(1,2) = this.i_row*this.panel_height+6 ;
            else
                corner_position(1,1) = 1 + (this.i_row-1)*this.panel_height+6;
                corner_position(1,2) = this.i_row*this.panel_height;
            end

            %get the width range of the panel
            if this.i_column == 1
                corner_position(2,1) = 1;
                corner_position(2,2) = this.panel_width-2;
            elseif this.i_column ~= this.n_panel_column
                corner_position(2,1) = (this.i_column-1)*this.panel_width+1-2;
                corner_position(2,2) = this.i_column*this.panel_width-2;
            else
                corner_position(2,1) = (this.i_column-1)*this.panel_width+1-2;
                corner_position(2,2) = 1996;
            end
            
            %update the iterator
            
            %if the iterator is on the bottom panel, move it to the top
            %panel and move to the right panel
            if this.i_row == 2
                this.i_row = 1;
                this.i_column = this.i_column + 1;
            %else the iterator is on the top panel, move it down
            else
                this.i_row = this.i_row + 1;
            end
            
        end %getNextPanelCorner(this)
        
    end
    
end

