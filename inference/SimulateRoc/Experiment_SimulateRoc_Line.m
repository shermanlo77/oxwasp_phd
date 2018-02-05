classdef Experiment_SimulateRoc_Line < Experiment_SimulateRoc
    
    properties
    end
    
    %METHODS
    methods (Access = public)
        
        %CONSTRUCTOR
        function this = Experiment_SimulateRoc_Line()
            this@Experiment_SimulateRoc('SimulateRoc_Line');
        end
        
        %OVERRIDE: PRINT RESULTS
        function printResults(this)
        end
        
    end
    
    %PROTECTED METHODS
    methods (Access = protected)
        
        %OVERRIDE: SETUP
        function setup(this)
            %call superclass version of setup
            this.setup@Experiment_SimulateRoc(RandStream('mt19937ar','Seed',uint32(3367688732)));
        end
        
        %IMPLEMENTED: GET DEFECT SIMULATOR
            %Return defect simulator which adds a plane and a line defect
            %parameter is the defect intensity
        function defect_simulator = getDefectSimulator(this, parameter)
            data = this.getData();
            defect_simulator = DefectSimulator([data.height,data.width]);
            defect_simulator.addPlane(3*[1;1]);
            defect_simulator.addLineDefect(round(data.width/2),5,parameter);
        end
        
    end
    
end

