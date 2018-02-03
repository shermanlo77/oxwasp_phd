%SUBCLASS: EXPERIMENT NO DEFECT WITH A SINUSOID
%See superclass Experiment_noDefect
%
%This implementation uses a sinusoid as a smooth function
classdef Experiment_noDefect_sinusoid < Experiment_noDefect
    
    properties
    end
    
    %METHODS
    methods (Access = public)
        
        %CONSTRUCTOR
        function this = Experiment_noDefect_sinusoid()
            this@Experiment_noDefect('noDefect_sinusoid');
        end
        
    end
    
    %PROTECTED METHODS
    methods (Access = protected)
        
        %OVERRIDE: SETUP
        function setup(this)
            %call superclass version of setup
            this.setup@Experiment_noDefect(RandStream('mt19937ar','Seed',uint32(707037501)), linspace(0,1E4,6));
        end
        
        %IMPLEMENTED: GET DEFECT SIMULATOR
            %Return defect simulator which adds a sinusoid, parameter is the amplitude
        function defect_simulator = getDefectSimulator(this, parameter)
            data = this.getData();
            defect_simulator = DefectSimulator([data.height,data.width]);
            defect_simulator.addSinusoid(parameter, [750;750],0);
        end
    end
    
end

