classdef Experiment_GLMVarMean_Sep16_30deg < Experiment_GLMVarMean_Sep16
    
    properties
    end
    
    methods
        
        %CONSTRUCTOR
        function this = Experiment_GLMVarMean_Sep16_30deg()
            %call superclass with experiment name
            this@Experiment_GLMVarMean_Sep16('GLMVarMean_Sep16_30deg');
        end
        
        %OVERRIDE: SET UP EXPERIMENT
        function setUpExperiment(this)
            %call superclass with 100 repeats and a random seed
            this.setUpExperiment@Experiment_GLMVarMean_Sep16(uint32(1562360917));
        end
        
        function scan = getScan(this)
            scan = AbsBlock_Sep16_30deg();
        end
    end
    
end

