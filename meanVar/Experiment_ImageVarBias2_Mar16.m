classdef Experiment_ImageVarBias2_Mar16 < Experiment_ImageVarBias2
    
    properties
    end
    
    methods
        
        %CONSTRUCTOR
        function this = Experiment_ImageVarBias2_Mar16()
            %call superclass with experiment name
            this@Experiment_ImageVarBias2('ImageVarBias2_Mar16');
        end
        
        %OVERRIDE: SET UP EXPERIMENT
        function setUpExperiment(this)
            %call superclass with 100 repeats and a random stream
            this.setUpExperiment@Experiment_ImageVarBias2(100, RandStream('mt19937ar','Seed',uint32(2133104798)));
        end
        
        %IMPLEMENTED: GET SCAN
        function scan = getScan(this)
            scan = AbsBlock_Mar16();
        end
        
        %IMPLEMENTED: GET N GLM
        function n_glm = getNGlm(this)
            n_glm = 10;
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
                    model = MeanVar_GLM(this.shape_parameter,-3,LinkFunction_Canonical());
                case 5
                    model = MeanVar_GLM(this.shape_parameter,-4,LinkFunction_Canonical());
                case 6
                    model = MeanVar_GLM(this.shape_parameter,1,LinkFunction_Log());
                case 7
                    model = MeanVar_GLM(this.shape_parameter,-1,LinkFunction_Log());
                case 8
                    model = MeanVar_GLM(this.shape_parameter,-2,LinkFunction_Log());
                case 9
                    model = MeanVar_GLM(this.shape_parameter,-3,LinkFunction_Log());
                case 10
                    model = MeanVar_kNN(1E3);
            end
        end
        
        %returns number of shading correctors to investigate
        function n_shad = getNShadingCorrector(this)
            n_shad = 1;
        end
        
        %returns shading corrector given index
        %index can range from 1 to getNShadingCorrector
        %RETURNS:
            %shading_corrector: ShadingCorrector object
            %reference_index: row vector containing integers
                %pointing to which reference scans to be used for shading correction training
        function [shading_corrector, reference_index] = getShadingCorrector(this, index)
            shading_corrector = ShadingCorrector_null();
            reference_index = [];
        end
        
    end
    
end

