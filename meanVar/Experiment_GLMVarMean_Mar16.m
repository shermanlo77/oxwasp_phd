%EXPERIMENT GLM VAR MEAN MAR 16
%See superclass Experiment_GLMVarMean
%For the dataset AbsBlock_Mar13, only top half of the image is used to avoid the foam
classdef Experiment_GLMVarMean_Mar16 < Experiment_GLMVarMean
    
    properties
    end
    
    methods
        
        %CONSTRUCTOR
        function this = Experiment_GLMVarMean_Mar16()
            %call superclass with experiment name
            this@Experiment_GLMVarMean('GLMVarMean_Mar16');
        end
        
        %OVERRIDE: SET UP EXPERIMENT
        function setUpExperiment(this)
            %call superclass with 100 repeats and a random stream
            this.setUpExperiment@Experiment_GLMVarMean(100, RandStream('mt19937ar','Seed',uint32(176048084)));
        end
        
        %GET SEGMENTATION
        %Return the binary segmentation image
        %true for pixels to be considered
        function segmentation = getSegmentation(this)
            scan = this.getScan();
            segmentation = scan.getSegmentation();
            segmentation((scan.height/2+1):end,:) = false;
        end
        
        %IMPLEMENTED: GET SCAN
        function scan = getScan(this)
            scan = AbsBlock_Mar16();
        end
        
        %IMPLEMENTED: GET N GLM
        function n_glm = getNGlm(this)
            n_glm = 9;
        end
        
        %IMPLEMENTED: GET GLM
        function model = getGlm(this, shape_parameter, index)
            switch index
                case 1
                    model = MeanVar_GLM(shape_parameter,1,LinkFunction_Identity());
                case 2
                    model = MeanVar_GLM(shape_parameter,-1,LinkFunction_Canonical());
                case 3
                    model = MeanVar_GLM(shape_parameter,-2,LinkFunction_Canonical());
                case 4
                    model = MeanVar_GLM(shape_parameter,-3,LinkFunction_Canonical());
                case 5
                    model = MeanVar_GLM(shape_parameter,-4,LinkFunction_Canonical());
                case 6
                    model = MeanVar_GLM(shape_parameter,1,LinkFunction_Log());
                case 7
                    model = MeanVar_GLM(shape_parameter,-1,LinkFunction_Log());
                case 8
                    model = MeanVar_GLM(shape_parameter,-2,LinkFunction_Log());
                case 9
                    model = MeanVar_GLM(shape_parameter,-3,LinkFunction_Log());
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

