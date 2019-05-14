%ABSTRACT CLASS: PANEL COUNTER
%Instantised objects from this class serve the purpose with return the coordinates of each panel in 
    %an iterative fashion
%To use this class:
  %instantise an object from this class
  %use this.hasNextPanelCorner to see if there are more panel coordinates to be obtaiend
  %use this.getNextPanelCorner to get the top-left and bottom right coordinates of this panel
      %once this method is called, the iterator will move to the next panel
      %the iterator should move row by row, then column by column (i.e. downwards)
classdef PanelCounter < handle

  %MEMBER VARIABLES
  properties
    panelHeight; %height of the panels
    panelWidth; %width of the panels
    nPanelColumn; %number of columns of panels
    nPanel; %number of panels

    iRow; %iterator, counting the number of panel columns
    iColumn; %iterator, counting the number of panel rows
  end

  %METHODS
  methods

    %CONSTRUCTOR
    function this = PanelCounter()
      %get the iterator member variables
      this.resetPanelCorner();
    end

    %METHOD: RESET PANEL CORNER
    %Reset the iteartor for obtaining the corners of the panel
    function resetPanelCorner(this)
      this.iRow = 1;
      this.iColumn = 1;
    end

    %METHOD: HAS NEXT PANEL CORNER
    %Output boolean, true if the iterator has another panel to iterate through
    function hasNext = hasNextPanelCorner(this)
      %if the iterator has counted beyound the number of columns
      if (this.iColumn > this.nPanelColumn)
        %the iterator has no more panels to iterator
        hasNext = false;
        %else it has, return true
      else
        hasNext = true;
      end
    end

  end

  %ABSTRACT METHODS
  methods (Abstract)

    %METHOD: GET NEXT PANEL CORNER
    %Get the top left and bottom right coordinates of the next panel in
    %the iterator
    %RETURN:
      %cornerPosition: 2x2 matrix, each column cotains the
      %coordinates of the top left and bottom right of the panel. The
      %coordinates are in the form of matrix index, i.e. 1st row is
      %for the height, 2nd row for the width.
    cornerPosition = getNextPanelCorner(this);

  end

end

