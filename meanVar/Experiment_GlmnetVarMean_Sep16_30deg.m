classdef Experiment_GlmnetVarMean_Sep16_30deg < Experiment_GlmnetVarMean_Sep16
    
    properties
    end
    
    methods
        
        %CONSTRUCTOR
        function this = Experiment_GlmnetVarMean_Sep16_30deg()
            %call superclass with experiment name
            this@Experiment_GlmnetVarMean_Sep16('GlmnetVarMean_Sep16_30deg');
        end
        
        %OVERRIDE: SET UP EXPERIMENT
        function setUpExperiment(this)
            %call superclass with 100 repeats and a random stream
            this.setUpExperiment@Experiment_GlmnetVarMean_Sep16(RandStream('mt19937ar','Seed',uint32(4060924453)));
        end
        
        function scan = getScan(this)
            scan = AbsBlock_Sep16_30deg();
        end
    end
    
end

