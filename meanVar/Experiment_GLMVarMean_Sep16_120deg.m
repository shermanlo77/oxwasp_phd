classdef Experiment_GLMVarMean_Sep16_120deg < Experiment_GLMVarMean_Sep16
    
    properties
    end
    
    methods
        
        %CONSTRUCTOR
        function this = Experiment_GLMVarMean_Sep16_120deg()
            %call superclass with experiment name
            this@Experiment_GLMVarMean_Sep16('GLMVarMean_Sep16_120deg');
        end
        
        %OVERRIDE: SET UP EXPERIMENT
        function setUpExperiment(this)
            %call superclass with 100 repeats and a random stream
            this.setUpExperiment@Experiment_GLMVarMean_Sep16(RandStream('mt19937ar','Seed',uint32(1277156729)));
        end
        
        function scan = getScan(this)
            scan = AbsBlock_Sep16_120deg();
        end
    end
    
end

