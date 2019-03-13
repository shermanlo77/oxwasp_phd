%PANEL COUNTER (BRASS SUBCLASS)
%Brass is just a random noun
%See superclass PanelCounter for derived member variables and methods
classdef PanelCounterBrass < PanelCounter
  
  %MEMBER VARIABLES
  properties
  end
  
  %METHODS
  methods
    
    %CONSTRUCTOR
    function this = PanelCounterBrass()
      this@PanelCounter(); %call superclass
      %assign derived member variables
      this.panelHeight = 998;
      this.panelWidth = 128;
      this.nPanelColumn = 16;
      this.nPanel = 32;
    end
    
    %IMPLMENTED: GET NEXT PANEL CORNER
    %Get the top left and bottom right coordinates of the next panel in the iterator
    %RETURN:
      %corner_position: 2x2 matrix, each column cotains the
      %coordinates of the top left and bottom right of the panel. The
      %coordinates are in the form of matrix index, i.e. 1st row is
      %for the height, 2nd row for the width.
    function cornerPosition = getNextPanelCorner(this)
      
      %declare 2x2 matrix
      cornerPosition = zeros(2,2);
      
      %get the height_range of the panel
      if this.iRow == 1
        cornerPosition(1,1) = 1 + (this.iRow-1)*this.panelHeight;
        cornerPosition(1,2) = this.iRow*this.panelHeight+6 ;
      else
        cornerPosition(1,1) = 1 + (this.iRow-1)*this.panelHeight+6;
        cornerPosition(1,2) = this.iRow*this.panelHeight;
      end
      
      %get the width range of the panel
      if this.iColumn == 1
        cornerPosition(2,1) = 1;
        cornerPosition(2,2) = this.panelWidth-2;
      elseif this.iColumn ~= this.n_panel_column
        cornerPosition(2,1) = (this.iColumn-1)*this.panelWidth+1-2;
        cornerPosition(2,2) = this.iColumn*this.panelWidth-2;
      else
        cornerPosition(2,1) = (this.iColumn-1)*this.panelWidth+1-2;
        cornerPosition(2,2) = 1996;
      end
      
      %update the iterator
      
      %if the iterator is on the bottom panel, move it to the top
      %panel and move to the right panel
      if this.iRow == 2
        this.iRow = 1;
        this.iColumn = this.iColumn + 1;
        %else the iterator is on the top panel, move it down
      else
        this.iRow = this.iRow + 1;
      end
      
    end %getNextPanelCorner(this)
    
  end
  
end

