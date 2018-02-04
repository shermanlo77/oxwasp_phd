%SUBCLASS: EXPERIMENT NO DEFECT WITH A PLANE
%See superclass Experiment_noDefect
%
%This implementation uses a plane as a smooth function
classdef Experiment_noDefect_plane < Experiment_noDefect
    
    properties
    end
    
    %METHODS
    methods (Access = public)
        
        %CONSTRUCTOR
        function this = Experiment_noDefect_plane()
            this@Experiment_noDefect('noDefect_plane');
        end
        
        %OVERRIDE: PRINT RESULTS
        function printResults(this)
            this.printResults@Experiment_noDefect('gradient');
        end
        
    end
    
    %PROTECTED METHODS
    methods (Access = protected)
        
        %OVERRIDE: SETUP
        function setup(this)
            %call superclass version of setup
            this.setup@Experiment_noDefect(RandStream('mt19937ar','Seed',uint32(2272397425)), linspace(0,7,10));
        end
        
        %IMPLEMENTED: GET DEFECT SIMULATOR
            %Return defect simulator which adds a plane, parameter is the gradient
        function defect_simulator = getDefectSimulator(this, parameter)
            data = this.getData();
            defect_simulator = DefectSimulator([data.height,data.width]);
            defect_simulator.addPlane( parameter*[1;1]);
        end
        
    end
    
end
