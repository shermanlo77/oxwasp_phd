%EXPERIMENT GLM VAR MEAN Sep 16
%See superclass Experiment_GLMVarMean
classdef Experiment_GlmnetVarMean_Sep16 < Experiment_GLMVarMean
    
    properties
        alpha;
        lambda_array;
        polynomial_array;
        parameter_array;
    end
    
    methods
        
        %CONSTRUCTOR
        function this = Experiment_GlmnetVarMean_Sep16(experiment_name)
            %call superclass with experiment name
            this@Experiment_GLMVarMean(experiment_name);
        end
        
        %OVERRIDE: SET UP EXPERIMENT
        function setUpExperiment(this, rand_stream)
            %call superclass with 100 repeats and a random stream
            this.setUpExperiment@Experiment_GLMVarMean(1, rand_stream);
            this.alpha = 0.99;
            this.lambda_array = 10.^linspace(-4,-2,30);
            this.polynomial_array = [1,-1,-2,-3,-4];
            this.parameter_array = zeros(numel(this.polynomial_array)+1,this.getNGlm());
        end
        
        %OVERIDE: DO ONE ITERATION OF EXPERIMENT
        function doIteration(this,i_glm,i_shad)
            %get the training and test mse
            [error_training, error_test, parameter] = this.trainingTestMeanVar(this.getGlm(this.shape_parameter, i_glm), i_shad);
            %save the training and test mse in the array
            this.training_error_array(this.i_repeat,i_glm,i_shad) = error_training;
            this.test_error_array(this.i_repeat,i_glm,i_shad) = error_test;
            %save the parameter
            this.parameter_array(:,i_glm) = parameter;
        end
        
        %IMPLEMENTED: GET N GLM
        function n_glm = getNGlm(this)
            n_glm = 30;
        end
        
        %IMPLEMENTED: GET GLM
        function model = getGlm(this, shape_parameter, index)
            model = MeanVar_ElasticNet(shape_parameter,this.polynomial_array,LinkFunction_Canonical(),this.alpha,this.lambda_array(index));
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
            scan = this.getScan();
            reference_white = scan.reference_white;
            shading_corrector = ShadingCorrector();
            reference_index = 1:reference_white;
        end
        
    end
    
    methods (Abstract)
        
        %returns scan object
        scan = getScan(this);
        
    end
    
end

