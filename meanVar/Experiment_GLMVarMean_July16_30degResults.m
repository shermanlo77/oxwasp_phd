classdef Experiment_GLMVarMean_July16_30degResults < Experiment_GLMVarMean_July16_30deg

    properties
    end
    
    methods
        
        function this = Experiment_GLMVarMean_July16_30degResults()
            this@Experiment_GLMVarMean_July16_30deg('GLMVarMean_July16_30degResults');
        end
        
        function setUpExperiment(this)
            experiment = this.loadResults(1);
            n_repeat = experiment.n_repeat;
            this.training_error_array = zeros(n_repeat,this.getNGlm(),this.getNShadingCorrector());
            this.test_error_array = zeros(n_repeat,this.getNGlm(),this.getNShadingCorrector()); 
           
            for i_shad = 1:this.getNShadingCorrector()
                experiment = this.loadResults(i_shad);
                this.training_error_array(:,:,i_shad) = experiment.training_error_array();
                this.test_error_array(:,:,i_shad) = experiment.test_error_array();
            end
        end
        
        function experiment = loadResults(this, index)
            switch index
                case 1
                    experiment = load('results/GLMVarMean_July16_30deg_null');
                case 2
                    experiment = load('results/GLMVarMean_July16_30deg_bw');
                case 3
                    experiment = load('results/GLMVarMean_July16_30deg_linear');
            end
            experiment = experiment.this;
        end
        
        function doExperiment(this)
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
                    reference_index = 1:scan.getNReference();
            end
        end
        
    end   
    
end

