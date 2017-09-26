classdef Experiment_ImageVarBias2_Sep16 < Experiment_ImageVarBias2
    
    properties
    end
    
    methods
        
        %CONSTRUCTOR
        function this = Experiment_ImageVarBias2_Sep16(experiment_name)
            %call superclass with experiment name
            this@Experiment_ImageVarBias2(experiment_name);
        end
        
        %OVERRIDE: SET UP EXPERIMENT
        function setUpExperiment(this, rand_stream)
            %call superclass with 100 repeats and a random stream
            this.setUpExperiment@Experiment_ImageVarBias2(1000, rand_stream);
        end
        
        %IMPLEMENTED: GET N GLM
        function n_glm = getNGlm(this)
            n_glm = 5;
        end
        
        %IMPLEMENTED: GET GLM
        function model = getGlm(this, index)
            switch index
                case 1
                    model = MeanVar_GLM(this.shape_parameter,1,LinkFunction_Identity());
                case 2
                    model = MeanVar_GLM(this.shape_parameter,-1,LinkFunction_Canonical());
                case 3
                    model = MeanVar_GLM(this.shape_parameter,-2,LinkFunction_Canonical());
                case 4
                    model = MeanVar_GLM(this.shape_parameter,1,LinkFunction_Log());
                case 5
                    model = MeanVar_GLM(this.shape_parameter,-1,LinkFunction_Log());
            end
        end
        
        %returns number of shading correctors to investigate
        function n_shad = getNShadingCorrector(this)
            n_shad = 3;
        end
        
        %returns shading corrector given index
        %index can range from 1 to getNShadingCorrector
        %RETURNS:
            %shading_corrector: ShadingCorrector object
            %reference_index: row vector containing integers
                %pointing to which reference scans to be used for shading correction training
        function [shading_corrector, reference_index] = getShadingCorrector(this, index)
            scan = this.getScan();
            reference_white = scan.reference_white;
            switch index
                case 1
                    shading_corrector = ShadingCorrector_null();
                    reference_index = [];
                case 2
                    shading_corrector = ShadingCorrector();
                    reference_index = [1,reference_white];
                case 3
                    shading_corrector = ShadingCorrector();
                    reference_index = 1:reference_white;
            end
        end
        
    end
    
    methods (Abstract)
        
        %returns scan object
        scan = getScan(this);
        
    end
    
end

