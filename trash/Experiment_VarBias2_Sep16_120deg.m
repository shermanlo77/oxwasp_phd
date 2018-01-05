classdef Experiment_VarBias2_Sep16_120deg < Experiment_VarBias2
    
    properties
    end
    
    methods
        
        function this = Experiment_VarBias2_Sep16_120deg()
            this@Experiment_VarBias2('VarBias2_Sep16_120deg');
        end
        
        function setUpExperiment(this)
           this.setUpExperiment@Experiment_VarBias2(1000, 100, RandStream('mt19937ar','Seed',uint32(4028339853))); 
        end
        
        function scan = getScan(this)
           scan = AbsBlock_Sep16_120deg();
           scan.addDefaultShadingCorrector();
        end
        
    end
    
end

