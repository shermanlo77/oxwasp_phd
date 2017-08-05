classdef Experiment_GLMVarMean_July16_120deg_bw < Experiment_GLMVarMean_July16_120deg
    
    properties
    end
    
    methods
        
        %CONSTRUCTOR
        function this = Experiment_GLMVarMean_July16_120deg_bw()
            %call superclass with experiment name
            this@Experiment_GLMVarMean_July16_120deg('GLMVarMean_July16_120deg_bw');
        end
        
        %OVERRIDE: SET UP EXPERIMENT
        function setUpExperiment(this)
            %call superclass with 100 repeats and a random seed
            this.setUpExperiment@Experiment_GLMVarMean_July16_120deg(uint32(1032494379));
        end
        
        function [shading_corrector, reference_index] = getShadingCorrector(this, index)
            scan = this.getScan();
            shading_corrector = ShadingCorrector();
            reference_index = [1,scan.reference_white];
        end
    end
    
end

