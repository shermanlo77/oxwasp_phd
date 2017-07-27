classdef Experiment_GLMMeanVarTrainingTest < Experiment
    %EXPERIMENT_GLM_MEANVAR_TRAININGTEST Assess the performance of GLM fit on mean var data
    %   The images are spilt into 2 parts, training and test. GLM iss used
    %   to model the variance and mean relationship, with variance as the
    %   response. The response is gamma randomlly distributed with known
    %   shape parameter.   
    %
    %   Only the top half of the images were used, this is to avoid the
    %   form. In addition, the images were thresholded to only consider
    %   pixels from the 3d printed sample.
    %
    %   The training set is used to train the glm, which is then used to
    %   predict the variance of the test set. The training and mean
    %   standarised residuals are plotted, that is the residual divided by
    %   the std of gamma.
    %
    %   Different shading corrections, polynomial orders and link functions
    %   were considered. The experiment was repeated by reassigning the
    %   training and test set.
    
    %MEMBER VARIABLES
    properties
        
        i_repeat; %number of iterations done
        n_repeat; %number of itereations to complete the experiment
        
        glm_array; %array of glm objects (see constructor)
        %array of training and test mse
            %dim 1: for each repeat
            %dim 2: for each glm
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
        function this = Experiment_GLMMeanVarTrainingTest()
            %call superclass
            this@Experiment('GLMMeanVarTrainingTest');
        end

        %DECLARE RESULT ARRAY
        %PARAMETERS:
        function setUpExperiment(this)
            
            this.i_repeat = 1;
            this.n_repeat = 100;
            %assign member variables
            this.rand_stream = RandStream('mt19937ar','Seed',uint32(176048084));
            this.n_train = 50;
            this.threshold = AbsBlock_Mar16.getThreshold_topHalf();
            
            %location of the data
            %shape parameter
            shape_parameter = (this.n_train-1)/2;
            
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
            
            %see member variables
            this.training_mse_array = zeros(this.n_repeat,9);
            this.test_mse_array = zeros(this.n_repeat,9);
        end
        
        
        %DO EXPERIMENT
        function doExperiment(this)
            
            %do n_repeat times
            while (this.i_repeat <= this.n_repeat)
            
                %use its random stream
                RandStream.setGlobalStream(this.rand_stream);
                %for each glm
                for i_glm = 1:9
                    %get the training and test mse
                    [mse_training, mse_test] = this.trainingTestMeanVar(this.glm_array{i_glm});
                    %save the training and test mse in the array
                    this.training_mse_array(this.i_repeat,i_glm) = mse_training;
                    this.test_mse_array(this.i_repeat,i_glm) = mse_test;
                end
                
                %print the progress
                this.printProgress(this.i_repeat / this.n_repeat);
                %increment i_repeat
                this.i_repeat = this.i_repeat + 1;
                %save the state of this experiment
                this.saveState();
                
            end
        end
        
        
        %TRAINING/TEST MEAN VAR Gets the training and test MSE when fitting and predicting the mean and variance relationship
        %PARAMETERS:
            %model: variance model object
        %RETURN:
            %mse_training
            %mse_test
        function [mse_training, mse_test] = trainingTestMeanVar(this, model)

                data = AbsBlock_Mar16();
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
                mse_training = model.getPredictionMSSE(sample_mean,sample_var);

                %get the variance mean data of the test set
                [sample_mean,sample_var] = data.getSampleMeanVar_topHalf(test_index);
                %segment the mean var data
                sample_mean(this.threshold) = [];
                sample_var(this.threshold) = [];
                %get the test mse
                mse_test = model.getPredictionMSSE(sample_mean,sample_var);

        end
        
        %PRINT RESULTS
        %Save the training and test MSE into a latex table
        function printResults(this)
            
            figure;
            boxplot(this.training_mse_array);
            
            figure;
            boxplot(this.test_mse_array);
            
        end
        
    end
    
end

