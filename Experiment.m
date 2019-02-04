%EXPERIMENT (Abstract) Superclass for check pointing experiments and saving results
%   Objects derived from this class has a name and an indicator if the experiment is completed
%   In addition, it is recommended the constructor of derived class should have no agurments
%
%   The instantised object is saved in a .mat file in the results folder, appended with the experiment name.
%   Call the method run() to run the experiment
%   
%   The constructor is designed to be run differently depending if the file exist
%   1. if there is no file, the constructor will call the method setup() save the instantised itself to a .mat file
%   2. if the .mat file can be loaded, the member variables are read from that and instantised with these member variables
%
%   Abstract methods:
%       setup() this is where ALL member variables from the derived class are assigned
%       doExperiment() run the full (or resume) the experiment
%       printResults() print the results using derived member variables
classdef Experiment < handle
    
    %MEMBER VARIABLES
    properties (SetAccess = protected)
        experiment_name; %string, name of the experiment and the file name for storing it in a .mat file
        is_complete; %boolean, true if the experiment is completed
    end
    properties (SetAccess = protected, GetAccess = protected)
        n_arrow; %number of arrows to be displayed in the progress bar
    end
    
    %METHODS
    methods (Access = public)
        
        %CONSTRUCTOR
        %PARAMETERS:
            %experiment_name: name of the experiment and the file name for storing it in a .mat file
        function this = Experiment(experiment_name)
            if nargin == 0
              experiment_name = class(this);
            end
            %try and load an existing .mat file, then either print and run the experiment
            try
                %load the file
                load(strcat('results/',experiment_name,'.mat'));
            %catch problems reading the file
            catch
                %assign member variables
                this.experiment_name = experiment_name;
                this.is_complete = false;
                this.n_arrow = -1;
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
            if this.is_complete
                disp(cell2mat({this.experiment_name,' already completed'}));
            %else, do the experiment, set is_complete to be true and save it
            else
                this.doExperiment();
                this.is_complete = true;
                this.save();
            end
        end
        
        function deleteResults(this)
          if (strcmp(input('Are you sure? ','s'),'yes'))
            delete(strcat('results/',this.experiment_name,'.mat'));
            disp(strcat('results/',this.experiment_name,'.mat deleted'));
          else
            disp('not deleted');
          end
        end
        
        %METHOD: SAVE STATE
        function save(this)
            save(strcat('results/',this.experiment_name,'.mat'),'this');
            %print text that the experiment has been saved
            disp(strcat('results/',this.experiment_name,'.mat saved'));
        end
        
    end %methods
    
    %PROTECTED METHODS
    methods (Access = protected)

        %METHOD: PRINT PROGRESS
        %Displays a progress bar (with resoloution of 20)
        %PARAMETERS:
            %p: fraction of progress done (between 0 and 1)
        function printProgress(this, p)

            %get the number of arrows to be displayed
            new_n_arrow = floor(p*20);
            
            %if the number of arrows to be displayed is bigger than the number of arrows displayed before
            if new_n_arrow > this.n_arrow

                %save the number of arrows
                this.n_arrow = new_n_arrow;
                
                %declare an array of . enclosed by square brackets
                progress_bar(1:22) = '.';
                progress_bar(1) = '[';
                progress_bar(end) = ']';

                %for each arrow, put an arrow
                for i = 1:this.n_arrow
                    progress_bar(i+1) = '>';
                end
                
                %display time
                time = clock;
                time = time(4:5);
                progress_bar = [progress_bar, '  Time:  ', num2str(time)];

                %display the progress bar
                disp(progress_bar);
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

