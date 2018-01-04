%EXPERIMENT (Abstract) Superclass for check pointing experiments and saving results
%   Objects derived from this class has a name and an indicator if the experiment is completed
%   In addition, it is recommended the constructor of derived class should have no agurments
%
%   The instantised object is saved in a .mat file in the results folder, appended with the experiment name.
%   
%   The constructor is designed to be run differently depending if the file exist
%   1. if there is no file, the constructor will call the method setup() save the instantised itself to a .mat file
%   2. if the .mat file can be loaded, the member variables are read from that and instantised with these member variables
%
%   Abstract methods:
%       setup() (this is where ALL member variables from the derived class are assigned)
%       run() (run the experiment, DON'T FORGET TO CALL this.saveState to save the member variables)
%       printResults (print the results using derived member variables)
classdef Experiment < handle
    
    %MEMBER VARIABLES
    properties (SetAccess = protected)
        experiment_name; %string, name of the experiment and the file name for storing it in a .mat file
        is_complete; %boolean, true if the experiment is completed
        n_arrow; %number of arrows to be displayed in the progress bar
    end
    
    %METHODS
    methods (Access = public)
        
        %CONSTRUCTOR
        %PARAMETERS:
            %experiment_name: name of the experiment and the file name for storing it in a .mat file
        function this = Experiment(experiment_name)
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
                this.saveState();
                %print text that the experiment has been saved
                disp(strcat('results/',this.experiment_name,'.mat saved'));
            end
        end %constructor
        
        %METHOD: SAVE STATE
        function saveState(this)
            save(strcat('results/',this.experiment_name,'.mat'),'this');
        end
        
        %METHOD: PRINT PROGRESS
        %Displays a progress bar (with resoloution of 20)
        %PARAMETERS:
            %p: fraction of progress done (between 0 and 1)
        function printProgress(this, p)

            %get the number of arrows to be displayed
            new_n_arrow = round(p*20);
            
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

                %display the progress bar
                disp(progress_bar);
            end

        end %printProgress
        
    end %methods
    
    %ABSTRACT METHODS
    methods (Abstract)
        
        %SETUP EXPERIMENT (Abstract method)
        setup(this)
        
        %DO EXPERIMENT (Abstract method)
        %Does the experiment
        run(this)
        
        %PRINT RESULTS (Abstract method)
        %Export the results to a figure or table, to be used by latex
        printResults(this)
         
    end
    
end

