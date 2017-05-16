%EXPERIMENT (Abstract) Superclass for check pointing experiments and saving results
%   Objects derived from this class has a name and an indicator if the experiment is completed
%   In addition, it is recommended the constructor of derived class should have no agurments
%   Results are to be stored in the member variables
%   
%   Results are save in a .mat file in the results folder, appended with the experiment name.
%   
%   The constructor is designed to be run differently depending if the file exist
%   1. if there is no file, the constructor save the instantised self to a .mat file
%   2. if the .mat file can be loaded but is_complete is false, it will run the experiment, change is_complete to true and save
%   3. if the .mat file can be loaded and is_complete is true, the results are printed
%
%   Abstract methods:
%       setUpExperiment (this is where ALL member variables from the derived class are assigned)
%       doExperiment (DON'T FORGET TO CALL this.saveState at every iteration!!!)
%       printResults (print the results using derived member variables)
classdef Experiment < handle
    
    %MEMBER VARIABLES
    properties
        experiment_name; %string, name of the experiment and the file name for storing it in a .mat file
        is_complete; %boolea, true if the experiment is completed
    end
    
    %METHODS
    methods
        
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
                %set up the experiment
                this.setUpExperiment;
                %save a .mat file and exit the constructor
                this.saveState();
                disp(strcat('results/',this.experiment_name,'.mat saved'));
                return;
            end
            
            %if the experiment is completed
            if this.is_complete
                %print the results
                this.printResults();
            %else the experiment is not completed
            else
                %print the name of the experiment and do the experiment
                disp(cell2mat(strcat({'Running ',this.experiment_name})));
                this.doExperiment();
                %change the boolean is_complete to true and save the state of the experiment
                this.is_complete = true;
                this.saveState();
            end
            
        end %constructor
        
        %FUNCTION: SAVE STATE
        function saveState(this)
            save(strcat('results/',this.experiment_name,'.mat'),'this');
        end
        
    end %methods
    
    %ABSTRACT METHODS
    methods (Abstract)
        
        %SETUP EXPERIMENT (Abstract method)
        setUpExperiment(this)
        
        %DO EXPERIMENT (Abstract method)
        %Does the experiment
        doExperiment(this)
        
        %PRINT RESULTS (Abstract method)
        %Export the results to a figure or table, to be used by latex
        printResults(this)
         
    end
    
    %STATIC METHODS
    methods (Static)
        
        %FUNCTION PRINT PROGRESS
        %Displays a progress bar (with resoloution of 20)
        %PARAMETERS:
            %p: fraction of progress done (between 0 and 1)
        function printProgress(p)

            %get the number of arrows to be displayed
            n_arrow = round(p*20);

            %declare an array of . enclosed by square brackets
            progress_bar(1:22) = '.';
            progress_bar(1) = '[';
            progress_bar(end) = ']';

            %for each arrow, put an arrow
            for i = 1:n_arrow
                progress_bar(i+1) = '>';
            end

            %display the progress bar
            disp(progress_bar);

        end %printProgress
                
    end %methods
    
end

