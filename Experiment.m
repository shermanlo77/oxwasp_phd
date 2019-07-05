%EXPERIMENT (Abstract) Superclass for check pointing experiments and saving results
%  Objects derived from this class has a name and an indicator if the experiment is completed
%  In addition, it is recommended the constructor of derived class should have no agurments
%
%  The instantised object is saved in a .mat file in the results folder, with the class name.
%  Call the method run() to run the experiment
%
%  The constructor is designed to be run differently depending if the file exist
%  1. if there is no file, the constructor will call the method setup() save the instantised itself
%      to a .mat file
%  2. if the .mat file can be loaded, the member variables are read from that and instantised with
%      these member variables
%
%  Abstract methods:
%    setup() this is where ALL member variables from the derived class are assigned
%    doExperiment() run the full (or resume) the experiment
%    printResults() print the results using derived member variables
classdef Experiment < handle
  
  %MEMBER VARIABLES
  properties (SetAccess = public)
    directory = fullfile('results'); %where the resuls are saved
    experimentName; %string, name of the experiment and the file name for storing it in a .mat file
    isComplete; %boolean, true if the experiment is completed
  end
  properties (SetAccess = protected, GetAccess = protected)
    nArrow; %number of arrows to be displayed in the progress bar
  end
  
  %METHODS
  methods (Access = public)
    
    %CONSTRUCTOR
    function this = Experiment()
      
      this.experimentName = class(this); %use the class as the experiment name
      
      %try and load an existing .mat file, then either print and run the experiment
      try
        %load the file
        load(fullfile(this.directory,strcat(this.experimentName,'.mat')));
        %catch problems reading the file
      catch
        %assign member variables
        this.isComplete = false;
        this.nArrow = -1;
        %set up the experiment
        this.setup();
        %save a .mat file
        this.save();
      end
    end %constructor
    
    %METHOD: RUN EXPERIMENT
    %Does the experiment if the experiment is not completed
    function run(this)
      %if the experiment is completed, throw an error
      if this.isComplete
        disp(cell2mat({this.experimentName,' already completed'}));
      %else, do the experiment, set isComplete to be true and save it
      else
        this.doExperiment();
        this.isComplete = true;
        this.save();
      end
    end
    
    %METHOD: DELETE RESULTS
    %Delete the .mat file storing the results
    function deleteResults(this)
      if (strcmp(input('Are you sure? ','s'),'yes'))
        delete(fullfile(this.directory,strcat(this.experimentName,'.mat')));
        disp(strcat(fullfile(this.directory,strcat(this.experimentName,'.mat')), ' deleted'));
      else
        disp('not deleted');
      end
    end
    
    %METHOD: SAVE STATE
    %Save itself in a .mat file
    function save(this)
      save(fullfile(this.directory,strcat(this.experimentName,'.mat')),'this');
      %print text that the experiment has been saved
      disp(strcat(fullfile(this.directory,strcat(this.experimentName,'.mat')), ' saved'));
    end
    
  end %methods
  
  %PROTECTED METHODS
  methods (Access = public)
    
    %METHOD: PRINT PROGRESS
    %Displays a progress bar (with resoloution of 20)
    %PARAMETERS:
    %p: fraction of progress done (between 0 and 1)
    function printProgress(this, p)
      
      %get the number of arrows to be displayed
      newNArrow = floor(p*20);
      
      %if the number of arrows to be displayed is bigger than the number of arrows displayed before
      if newNArrow > this.nArrow
        
        %save the number of arrows
        this.nArrow = newNArrow;
        
        %declare an array of . enclosed by square brackets
        progressBar(1:22) = '.';
        progressBar(1) = '[';
        progressBar(end) = ']';
        
        %for each arrow, put an arrow
        for i = 1:this.nArrow
          progressBar(i+1) = '>';
        end
        
        %display time
        time = clock;
        time = time(4:5);
        progressBar = [progressBar, ' ', class(this), ' at time: ', num2str(time)];
        
        %display the progress bar
        disp(progressBar);
      end
      
    end %printProgress
  end
  
  %ABSTRACT PROTECTED METHODS
  methods (Abstract, Access = protected)
    
    %SETUP EXPERIMENT (Abstract method)
    %Recommended to be protected
    setup(this)
    
    %DO EXPERIMENT (Abstract method)
    %Does (or resume) the experiment
    %Recommended to be protected
    doExperiment(this)
    
  end
  
  %ABSTRACT PUBLIC METHODS
  methods (Abstract, Access = public)
    
    %PRINT RESULTS (Abstract method)
    %Export the results to a figure or table, to be used by latex
    %Recommended to be protected
    printResults(this)
    
  end
end

