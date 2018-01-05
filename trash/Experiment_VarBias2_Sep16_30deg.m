classdef Experiment_VarBias2_Sep16_30deg < Experiment_VarBias2
    
    properties
    end
    
    methods
        
        function this = Experiment_VarBias2_Sep16_30deg()
            this@Experiment_VarBias2('VarBias2_Sep16_30deg');
        end
        
        function setUpExperiment(this)
           this.setUpExperiment@Experiment_VarBias2(1000, 100, RandStream('mt19937ar','Seed',uint32(3653123410))); 
        end
        
        function scan = getScan(this)
           scan = AbsBlock_Sep16_30deg();
           scan.addDefaultShadingCorrector();
        end
        
    end
    
end

