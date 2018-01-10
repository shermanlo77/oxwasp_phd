classdef Experiment_GlmDeviance < Experiment_GLMVarMean
    
    properties
        
        %array of deviance
            %dim 1: for each repeat
            %dim 2: for each glm
            %dim 3: for each shading corrector
        deviance_array;
        
    end
    
    methods (Access = public)
        
        %CONSTRUCTOR
        %PARAMETERS:
            %experiment_name
        function this = Experiment_GlmDeviance(experiment_name)
            %call superclass
            this@Experiment_GLMVarMean(experiment_name);
        end
        
    end
    
    methods (Access = protected)
        
        %OVERRIDE: SETUP
        %PARAMETERS:
            %n_repeat: number of times to repeat the experiment
            %rand_steam: random stream
        function setup(this, n_repeat, rand_stream)
            this.setup@Experiment_GLMVarMean(n_repeat, rand_stream);
            this.shape_parameter = (this.n_sample-1)/2;
        end
        
        %OVERRIDE: ASSIGN ARRAY
        function assignArray(this)
            this.deviance_array = zeros(this.n_repeat,this.getNGlm(),this.getNShadingCorrector());
        end
        
        %DO ONE ITERATION OF EXPERIMENT
        function doIteration(this)
            %get the model
            model = this.getGlm(this.i_glm);

            %get bootstrap index of the training and test data
            index_bootstrap = randi([1,this.n_sample],this.n_sample,1);

            %get variance mean data of the training set
            [sample_mean,sample_var] = this.getMeanVar(index_bootstrap);

            %train the classifier
            model.train(sample_mean,sample_var);
            
            %save the scaled deviance
            this.deviance_array(this.i_repeat,this.i_glm,this.i_shad) = model.scaled_deviance;
        end
        
    end
    
end

