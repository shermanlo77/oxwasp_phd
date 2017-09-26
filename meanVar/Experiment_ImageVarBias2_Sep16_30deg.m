classdef Experiment_ImageVarBias2_Sep16_30deg < Experiment_ImageVarBias2_Sep16
    
    properties
    end
    
    methods
        
        %CONSTRUCTOR
        function this = Experiment_ImageVarBias2_Sep16_30deg()
            %call superclass with experiment name
            this@Experiment_ImageVarBias2_Sep16('ImageVarBias2_Sep16_30deg');
        end
        
        %OVERRIDE: SET UP EXPERIMENT
        function setUpExperiment(this)
            %call superclass with 100 repeats and a random stream
            this.setUpExperiment@Experiment_ImageVarBias2_Sep16(RandStream('mt19937ar','Seed',uint32(3708793330)));
        end
        
        function scan = getScan(this)
            scan = AbsBlock_Sep16_30deg();
        end
    end
    
end

