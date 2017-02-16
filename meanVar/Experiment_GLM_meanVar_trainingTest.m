classdef Experiment_GLM_meanVar_trainingTest < Experiment
    %EXPERIMENT_GLM_MEANVAR_TRAININGTEST Summary of this class goes here
    %   Detailed explanation goes here
    
    %MEMBER VARIABLES
    properties
        
        block_array; %array of block images (shading uncorrected, b/w shading corrected, b/g/w shading corrected)
        glm_array; %array of glm objects (see constructor)
        %array of training and test mse
            %dim 1: for each repeat
            %dim 2: for each glm
            %dim 3: for each block
        training_mse_array;
        test_mse_array;
        %number of images in the training set
        n_train;
        %threshold logical image
        threshold;
        %random stream
        rand_stream;
        
    end
    
    %METHODS
    methods
        
        %CONSTRUCTOR
        function this = Experiment_GLM_meanVar_trainingTest()
            %call superclass
            this@Experiment('GLM_meanVar_trainingTest');
            %assign member variables
            this.rand_stream = RandStream('mt19937ar','Seed',uint32(176048084));
            this.n_train = 50;
            this.threshold = BlockData_140316.getThreshold_topHalf();
            
            %location of the data
            block_location = '../data/140316';
            %shape parameter
            shape_parameter = (this.n_train-1)/2;
            
            %declare array of block images (3 of them)
            this.block_array = cell(3,1);
            %1st one has no shading correction
            this.block_array{1} = BlockData_140316(block_location);
            %2nd one uses b/w for shading correction
            this.block_array{2} = BlockData_140316(block_location);
            this.block_array{2}.addShadingCorrector(@ShadingCorrector,false);
            %3rd one uses b/g/w for shading correction
            this.block_array{3} = BlockData_140316(block_location);
            this.block_array{3}.addShadingCorrector(@ShadingCorrector,true);
            
            %declare array of glm (with different link functions and polynomial feature)
            this.glm_array = cell(9,1);
            %identity link with polynomial order 1
            this.glm_array{1} = MeanVar_GLM_identity(shape_parameter,1);
            %canonical link with polynomial orders -1, -2, -3, -4
            this.glm_array{2} = MeanVar_GLM_canonical(shape_parameter,-1);
            this.glm_array{3} = MeanVar_GLM_canonical(shape_parameter,-2);
            this.glm_array{4} = MeanVar_GLM_canonical(shape_parameter,-3);
            this.glm_array{5} = MeanVar_GLM_canonical(shape_parameter,-4);
            %log link with polynomial orders -1, -2, -3, -4
            this.glm_array{6} = MeanVar_GLM_log(shape_parameter,-1);
            this.glm_array{7} = MeanVar_GLM_log(shape_parameter,-2);
            this.glm_array{8} = MeanVar_GLM_log(shape_parameter,-3);
            this.glm_array{9} = MeanVar_GLM_log(shape_parameter,-4);
            
        end

        %DECLARE RESULT ARRAY
        %PARAMETERS:
            %n_repeat: number of times to repeat the experiment
        function declareResultArray(this,n_repeat)
            %see member variables
            this.training_mse_array = zeros(n_repeat,9,3);
            this.test_mse_array = zeros(n_repeat,9,3);
        end
        
        
        %DO EXPERIMENT
        function doExperiment(this)
            %use its random stream
            RandStream.setGlobalStream(this.rand_stream);
            %for each block
            for i_block = 1:3
                %for each glm
                for i_glm = 1:9
                    %get the training and test mse
                    [mse_training, mse_test] = this.trainingTestMeanVar(this.block_array{i_block}, this.glm_array{i_glm});
                    %save the training and test mse in the array
                    this.training_mse_array(this.i_repeat,i_glm,i_block) = mse_training;
                    this.test_mse_array(this.i_repeat,i_glm,i_block) = mse_test;
                end
            end
        end
        
        
        %TRAINING/TEST MEAN VAR Gets the training and test MSE when fitting and predicting the mean and variance relationship
        %PARAMETERS:
            %data: data object
            %model: variance model object
        %RETURN:
            %mse_training
            %mse_test
        function [mse_training, mse_test] = trainingTestMeanVar(this, data, model)

                %get random index of the training and test data
                index_suffle = randperm(data.n_sample);
                training_index = index_suffle(1:this.n_train);
                test_index = index_suffle((this.n_train+1):data.n_sample);

                %get variance mean data of the training set
                [sample_mean,sample_var] = data.getSampleMeanVar_topHalf(training_index);
                %segment the mean var data
                sample_mean(this.threshold) = [];
                sample_var(this.threshold) = [];

                %train the classifier
                model.train(sample_mean,sample_var);
                %get the training mse
                mse_training = model.getPredictionMSE(sample_mean,sample_var);

                %get the variance mean data of the test set
                [sample_mean,sample_var] = data.getSampleMeanVar_topHalf(test_index);
                %segment the mean var data
                sample_mean(this.threshold) = [];
                sample_var(this.threshold) = [];
                %get the test mse
                mse_test = model.getPredictionMSE(sample_mean,sample_var);

        end
        
    end
    
    methods(Static)
        
        %GLOBAL: Call this to start experiment automatically
        function GLOBAL()
            %repeat the experiment this many times
            n_repeat = 100;
            %set up the experiment
            Experiment.setUpExperiment(@Experiment_GLM_meanVar_trainingTest,n_repeat);
            %run the experiment
            Experiment.runExperiments('GLM_meanVar_trainingTest',n_repeat);
        end
        
    end
    
end

