classdef Experiment_GlmnetVarMean_Sep16_120deg < Experiment_GlmnetVarMean_Sep16
    
    properties
    end
    
    methods
        
        %CONSTRUCTOR
        function this = Experiment_GlmnetVarMean_Sep16_120deg()
            %call superclass with experiment name
            this@Experiment_GlmnetVarMean_Sep16('GlmnetVarMean_Sep16_120deg');
        end
        
        %OVERRIDE: SET UP EXPERIMENT
        function setup(this)
            %call superclass with 100 repeats and a random stream
            this.setup@Experiment_GlmnetVarMean_Sep16(RandStream('mt19937ar','Seed',uint32(1673964387)));
        end
        
        function scan = getScan(this)
            scan = AbsBlock_Sep16_120deg();
        end
    end
    
end

