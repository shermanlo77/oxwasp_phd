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
                    experiment = strcat('results/',Experiment_GLMVarMean_July16_30deg_null.getExperimentName());
                case 2
                    experiment = strcat('results/',Experiment_GLMVarMean_July16_30deg_bw.getExperimentName());
                case 3
                    experiment = strcat('results/',Experiment_GLMVarMean_July16_30deg_linear.getExperimentName());
            end
            experiment = load(experiment);
            experiment = experiment.this;
        end
        
        function doExperiment(this)
        end

        %returns number of shading correctors to investigate
        function n_shad = getNShadingCorrector(this)
            n_shad = this.getNShadingCorrector@Experiment_GLMVarMean_July16();
        end  
        
    end
    
end

