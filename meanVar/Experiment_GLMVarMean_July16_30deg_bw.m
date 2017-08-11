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
            this.i_shading_correction = 2;
            %call superclass with 100 repeats and a random seed
            this.setUpExperiment@Experiment_GLMVarMean_July16_30deg(RandStream.create('mrg32k3a','Seed',uint32(3367632315),'NumStreams',Experiment_GLMVarMean_July16.getNShadingCorrector(),'StreamIndices',this.i_shading_correction));
        end
        
        function [shading_corrector, reference_index] = getShadingCorrector(this, ~)
            [shading_corrector, reference_index] = this.getShadingCorrector@Experiment_GLMVarMean_July16(this.i_shading_correction);
        end
    end
    
    methods (Static)
        
        function experiment_name = getExperimentName()
            experiment_name = 'GLMVarMean_July16_30deg_bw';
        end
        
    end
    
end

