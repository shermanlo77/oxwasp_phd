classdef Experiment < handle
    %EXPERIMENT (Abstract) Superclass for check pointing experiments and saving
    %results
    %   Objects derived from this class has a name, random number generator
    %   and a counter for counting the number of times the experiment was
    %   repeated. The method doExperiment does one iteration of an
    %   experiment.
    %
    %   The static method setUpExperiment help setup and save an experiment
    %   in a .mat file in the results folder. The runExperiments method
    %   then runs the experiment for a requested number of times.
    
    %MEMBER VARIABLES
    properties
        experiment_name; %string, name of the experiment and the file name for storing it in a .mat file
        i_repeat; %how many times the experiment was repeated + 1
    end
    
    %METHODS
    methods
        
        %CONSTRUCTOR
        %PARAMETERS:
            %experiment_name: name of the experiment and the file name for storing it in a .mat file
        function this = Experiment(experiment_name)
            %assign member variables
            this.experiment_name = experiment_name;
            this.i_repeat = 1;
        end
        
    end
    
    %ABSTRACT METHODS
    methods (Abstract)
        
        %DECLARE RESULT ARRAY (Abstract method)
        %Declare an array for storing results in its member variables
        %PARAMETERS:
            %n_repeat: number of times the experiment is to be repeated
        declareResultArray(this,n_repeat)
        
        %DO EXPERIMENT (Abstract method)
        %Does one iteration of the experiment, save the results
        %to its member variables
        doExperiment(this)
        
        %PRINT RESULTS (Abstract method)
        %Export the results to a figure or table, to be used by latex
        printResults(this)
         
    end
    
    %STATIC METHODS
    methods (Static)
        
        %SET UP EXPERIMENT
        %Instantise an experiment and saves it to a .mat file in the
        %results folder
        %PARAMETERS:
            %experiment_name: name of the experiment, this will be the name of the .mat file which saves the experiment object
            %experiment_handle: function handle for instantiating an experiment object
            %n_repeat: number of times to repeat the experiment in the plan
        function setUpExperiment(experiment_handle,n_repeat)
            %instantise an experiment object
            experiment = feval(experiment_handle);
            %try loading the file storing the .mat file
            try
                %load the file
                load(strcat('results/',experiment.experiment_name,'.mat'));
                %the file has already been saved return error 
                warning('Experiment already set up');
            %catch problems reading the file
            catch
                %declare an array of results in the experiment
                experiment.declareResultArray(n_repeat);
                %save the experiment object
                save(strcat('results/',experiment.experiment_name,'.mat'),'experiment');
            end
        end
        
        %RUN EXPERIMENTS
        %Run/continue the experiment many times till n_repeat iterations have been
        %completed
        %PARAMETERS:
            %experiment_name: name of the experiment
            %n_repat: number of iterations to be completed so far
        function runExperiments(experiment_name,n_repeat)
            
            %load the experiment object
            load_data = load(strcat('results/',experiment_name,'.mat'));
            experiment = load_data.experiment;
            
            %set up progress par
            h = waitbar(0,cell2mat(strcat({'Running experiment: '},experiment_name)));
            
            %while there are experiments to be done
            while experiment.i_repeat <= n_repeat
                %do the experiment
                experiment.doExperiment();
                %increment the member variable n_repeat
                experiment.i_repeat = experiment.i_repeat + 1;
                %save the experiment
                save(strcat('results/',experiment_name,'.mat'),'experiment');
                %move progress bar
                waitbar((experiment.i_repeat-1)/n_repeat);
            end
            
            %delete progress bar
            delete(h);
            
            %print the results
            experiment.printResults();
            
        end
                
    end
    
end

