%EXPERIMENT GLM VAR MEAN July 16
%See superclass Experiment_GLMVarMean
classdef Experiment_GlmDeviance_July16 < Experiment_GlmDeviance
    
    properties
    end
    
    methods (Access = public)
        
        %CONSTRUCTOR
        function this = Experiment_GlmDeviance_July16(experiment_name)
            %call superclass with experiment name
            this@Experiment_GlmDeviance(experiment_name);
        end
        
    end
    
    methods (Access = protected)
        
        %OVERRIDE: SET UP EXPERIMENT
        function setup(this, rand_stream)
            %call superclass with 100 repeats and a random stream
            this.setup@Experiment_GlmDeviance(100, rand_stream);
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
            n_shad = 3;
        end 

        %IMPLEMENTED: GET SHADING CORRECTOR
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
                    reference_index = 1:scan.reference_white;
            end
        end
        
    end
    
    methods (Abstract, Access = protected)
        
        %returns scan object
        scan = getScan(this);
        
    end   
    
end

