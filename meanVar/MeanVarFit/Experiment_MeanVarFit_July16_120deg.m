classdef Experiment_MeanVarFit_July16_120deg < Experiment_MeanVarFit_July16
    
    properties
    end
    
    methods (Access = public)
        
        %CONSTRUCTOR
        function this = Experiment_MeanVarFit_July16_120deg()
            %call superclass with experiment name
            this@Experiment_MeanVarFit_July16('MeanVarFit_July16_120deg');
        end
        
    end
    
    methods (Access = protected)
        
        %IMPLEMENTED: GET N GLM
        function n_glm = getNGlm(this)
            n_glm = 6;
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
                    model = KernelRegression(EpanechnikovKernel(),1E3);
            end
        end
        
        %IMPLEMENTED: GET SCAN
        function scan = getScan(this)
            scan = AbsBlock_July16_120deg();
        end

    end
    
end

