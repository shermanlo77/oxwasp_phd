classdef Experiment_GLMVarMean_July16_120deg < Experiment_GLMVarMean_July16
    
    properties
    end
    
    methods
        
        %CONSTRUCTOR
        function this = Experiment_GLMVarMean_July16_120deg(experiment_name)
            %call superclass with experiment name
            this@Experiment_GLMVarMean_July16(experiment_name);
        end
        
        %OVERRIDE: SET UP EXPERIMENT
        function setUpExperiment(this, unit32_seed)
            %call superclass with 100 repeats and a random seed
            this.setUpExperiment@Experiment_GLMVarMean_July16(unit32_seed);
        end
        
        function scan = getScan(this)
            scan = AbsBlock_July16_120deg();
        end
    end
    
    methods (Abstract)
        
        
        %returns shading corrector given index
        %index can range from 1 to getNShadingCorrector
        %RETURNS:
            %shading_corrector: ShadingCorrector object
            %reference_index: row vector containing integers
                %pointing to which reference scans to be used for shading correction training
        [shading_corrector, reference_index] = getShadingCorrector(this, index);
        
    end
    
end

