classdef Experiment_GLMVarMean_July16_30deg_bw < Experiment_GLMVarMean_July16_30deg
    
    properties
    end
    
    methods
        
        %CONSTRUCTOR
        function this = Experiment_GLMVarMean_July16_30deg_bw()
            %call superclass with experiment name
            this@Experiment_GLMVarMean_July16_30deg(Experiment_GLMVarMean_July16_30deg_bw.getExperimentName());
        end
        
        %OVERRIDE: SET UP EXPERIMENT
        function setUpExperiment(this)
            %call superclass with 100 repeats and a random seed
            this.setUpExperiment@Experiment_GLMVarMean_July16_30deg(uint32(1379936953));
        end
        
        function [shading_corrector, reference_index] = getShadingCorrector(this, index)
            scan = this.getScan();
            shading_corrector = ShadingCorrector();
            reference_index = [1,scan.reference_white];
        end
    end
    
    methods (Static)
        
        function experiment_name = getExperimentName()
            experiment_name = 'GLMVarMean_July16_30deg_bw';
        end
        
    end
    
end

