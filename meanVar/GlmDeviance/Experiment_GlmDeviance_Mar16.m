%EXPERIMENT GLM VAR MEAN MAR 16
%See superclass Experiment_GLMVarMean
%For the dataset AbsBlock_Mar13, only top half of the image is used to avoid the foam
classdef Experiment_GlmDeviance_Mar16 < Experiment_GlmDeviance
    
    properties
    end
    
    %METHODS
    methods (Access = public)
        
        %CONSTRUCTOR
        function this = Experiment_GlmDeviance_Mar16()
            %call superclass with experiment name
            this@Experiment_GlmDeviance('GlmDeviance_Mar16');
        end
        
    end
    
    %PROTECTED METHODS
    methods (Access = protected)
        
        %OVERRIDE: SET UP EXPERIMENT
        function setup(this)
            %call superclass with 100 repeats and a random stream
            this.setup@Experiment_GLMVarMean(100, RandStream('mt19937ar','Seed',uint32(2034516703)));
        end
        
        %IMPLEMENTED: GET SCAN
        function scan = getScan(this)
            scan = AbsBlock_Mar16();
        end
        
        %IMPLEMENTED: GET N GLM
        function n_glm = getNGlm(this)
            n_glm = 5;
        end
        
        %IMPLEMENTED: GET GLM
        function model = getGlm(this, index)
            switch index
                case 1
                    model = GlmGamma(1,IdentityLink());
                    model.setShapeParameter(this.shape_parameter);
                case 2
                    model = GlmGamma(-1,InverseLink());
                    model.setShapeParameter(this.shape_parameter);
                case 3
                    model = GlmGamma(-2,InverseLink());
                    model.setShapeParameter(this.shape_parameter);
                case 4
                    model = GlmGamma(1,LogLink());
                    model.setShapeParameter(this.shape_parameter);
                case 5
                    model = GlmGamma(-1,LogLink());
                    model.setShapeParameter(this.shape_parameter);
            end
        end
        
        %IMPLEMENTED: GET NUMBER OF SHADING CORRECTORS
        %returns number of shading correctors to investigate
        function n_shad = getNShadingCorrector(this)
            n_shad = 1;
        end
        
        %IMPLEMENTED: GET SHADING CORRECTOR
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

