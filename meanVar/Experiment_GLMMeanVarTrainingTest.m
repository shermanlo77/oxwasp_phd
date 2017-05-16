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
            this.threshold = BlockData_140316.getThreshold_topHalf();
            
            %location of the data
            block_location = 'data/140316';
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
            
            %when using shading correction, turn on setting extreme greyvalues to NaN and to remove dead pixels
            %NaN pixels are treated as dead
            for i_block = 2:3
                this.block_array{i_block}.turnOnSetExtremeToNan();
                this.block_array{i_block}.turnOnRemoveDeadPixels();
            end
            
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
            this.training_mse_array = zeros(this.n_repeat,9,3);
            this.test_mse_array = zeros(this.n_repeat,9,3);
        end
        
        
        %DO EXPERIMENT
        function doExperiment(this)
            
            %do n_repeat times
            while (this.i_repeat <= this.n_repeat)
            
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
            
            %training_table is an array of strings (10 x 5 size), to be
            %exported into a latex table
            training_table = string();
            training_table(10,5) = string();
            
            %1st row is the header
            training_table(1,1) = 'Link function';
            training_table(1,2) = 'Polynomial order';
            training_table(1,3) = 'Shading uncorrected';
            training_table(1,4) = 'Shading corrected BW';
            training_table(1,5) = 'Shading corrected BGW';
            
            %the test table has the same headers as the training table
            test_table = training_table;

            %for each glm
            for i_glm = 1:9
                
                %put the link function in the 1st column
                if isa(this.glm_array{i_glm},'MeanVar_GLM_identity')
                    training_table(i_glm+1,1) = 'Identity';
                elseif isa(this.glm_array{i_glm},'MeanVar_GLM_canonical')
                    training_table(i_glm+1,1) = 'Canonical';
                elseif isa(this.glm_array{i_glm},'MeanVar_GLM_log')
                    training_table(i_glm+1,1) = 'Log';
                end
                
                %put the polynomial order in the 2nd column
                training_table(i_glm+1,2) = num2str(this.glm_array{i_glm}.polynomial_order);
                
                %the link function and polynomial order are the same in the
                %training and test table
                test_table(i_glm+1,1:2) = training_table(i_glm+1,1:2);
                
                %for each shading correction
                for i_shading = 1:3
                    
                    %get the training mse
                    training_mse_i = this.training_mse_array(:,i_glm,i_shading);
                    %if any of the training mse has nan, output nan
                    if any(isnan(training_mse_i))
                        training_table(i_glm+1,2+i_shading) = 'NaN';
                    %else all training mse isn't nan, quote the quartile of the training mse
                    else
                        training_table(i_glm+1,2+i_shading) = quoteQuartileError(training_mse_i,1);
                    end

                    %get the test mse
                    test_mse_i = this.test_mse_array(:,i_glm,i_shading);
                    %if any of the test mse has nan, output nan
                    if any(isnan(test_mse_i))
                        test_table(i_glm+1,2+i_shading) = 'NaN';
                    %else all test mse isn't nan, quote the quartile of the test mse
                    else
                        test_table(i_glm+1,2+i_shading) = quoteQuartileError(test_mse_i,1);
                    end
                end

            end

            %output the training and test table to a latex table
            printStringArrayToLatexTable(training_table, strcat('reports/tables/',this.experiment_name,'_training.tex_table'));
            printStringArrayToLatexTable(test_table, strcat('reports/tables/',this.experiment_name,'_test.tex_table'));
            
        end
        
    end
    
end

