classdef Experiment_GLMVarMean_July16_120deg_null < Experiment_GLMVarMean_July16_120deg
    
    properties
    end
    
    methods
        
        %CONSTRUCTOR
        function this = Experiment_GLMVarMean_July16_120deg_null()
            %call superclass with experiment name
            this@Experiment_GLMVarMean_July16_120deg('GLMVarMean_July16_120deg_null');
        end
        
        %OVERRIDE: SET UP EXPERIMENT
        function setUpExperiment(this)
            %call superclass with 100 repeats and a random seed
            this.setUpExperiment@Experiment_GLMVarMean_July16_120deg(uint32(2623555944));
        end
        
        function [shading_corrector, reference_index] = getShadingCorrector(this, index)
            shading_corrector = ShadingCorrector_null();
            reference_index = [];
        end
    end
    
end

