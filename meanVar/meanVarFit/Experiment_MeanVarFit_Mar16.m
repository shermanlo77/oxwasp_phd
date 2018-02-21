classdef Experiment_MeanVarFit_Mar16 < Experiment_MeanVarFit
    
    properties
    end
    
    %METHODS
    methods (Access = public)
        
        %CONSTRUCTOR
        function this = Experiment_MeanVarFit_Mar16()
            %call superclass with experiment name
            this@Experiment_MeanVarFit('MeanVarFit_Mar16');
        end
        
    end
    
    %PROTECTED METHODS
    methods (Access = protected)
        
        %IMPLEMENTED: GET SCAN
        function scan = getScan(this)
            scan = AbsBlock_Mar16();
        end
        
        %IMPLEMENTED: GET N GLM
        function n_glm = getNGlm(this)
            n_glm = 9;
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
                case 6
                    model = KernelRegression(EpanechnikovKernel(),1E1);
                case 7
                    model = KernelRegression(EpanechnikovKernel(),1E2);
                case 8
                    model = KernelRegression(EpanechnikovKernel(),1E3);
                case 9
                    model = KernelRegression(EpanechnikovKernel(),1E4);
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

