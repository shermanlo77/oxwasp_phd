classdef Experiment_GLMVarMean < Experiment
    %EXPERIMENT_GLMVARMEAN Assess the performance of GLM fit on mean var data
    %   The images are spilt into 2 parts, training and test. GLM is used
    %   to model the variance and mean relationship, with variance as the
    %   response. The response is gamma randomlly distributed with known
    %   shape parameter.   
    %
    %   The images were segmented to only consider pixels from the ROI.
    %
    %   The training set is used to train the glm, which is then used to
    %   predict the variance of the test set. The training and mean
    %   standarised residuals are plotted, that is the residual divided by
    %   the std of gamma.
    
    %MEMBER VARIABLES
    properties
        
        i_repeat; %number of iterations done
        n_repeat; %number of itereations to complete the experiment
        n_sample; %number of images in a scan
        n_train; %number of images in the training set (half of n_sample)
        shape_parameter; %shape parameter of gamma
        
        %array of training and test error
            %dim 1: for each repeat
            %dim 2: for each glm
            %dim 3: for each shading corrector
        training_msse_array;
        test_msse_array;
        training_mse_array;
        test_mse_array;
        training_var_array;
        test_var_array;
        training_bias2_array;
        test_bias2_array;
        glm_array;
        
        training_index_array;
        test_index_array;
        
        %segmentation boolean vectpr
        segmentation;
        %random stream
        rand_stream;
           
    end
    
    %METHODS
    methods
        
        %CONSTRUCTOR
        %PARAMETERS:
            %experiment_name
        function this = Experiment_GLMVarMean(experiment_name)
            %call superclass
            this@Experiment(experiment_name);
        end

        %SET UP EXPERIMENT
        %PARAMETERS:
            %n_repeat: number of times to repeat the experiment
            %rand_steam: random stream
        function setUpExperiment(this, n_repeat, rand_stream)
            
            %get the scan object
            scan = this.getScan();
            
            %assign member variables
            this.i_repeat = 1;
            this.n_repeat = n_repeat;
            this.n_sample = scan.n_sample;
            this.n_train = round(this.n_sample/2);
            this.shape_parameter = (this.n_train-1)/2;
            
            this.training_msse_array = zeros(this.n_repeat,this.getNGlm(),this.getNShadingCorrector());
            this.test_msse_array = zeros(this.n_repeat,this.getNGlm(),this.getNShadingCorrector());
            
            this.training_mse_array = zeros(this.n_repeat,this.getNGlm(),this.getNShadingCorrector());
            this.test_mse_array = zeros(this.n_repeat,this.getNGlm(),this.getNShadingCorrector());
            
            this.training_var_array = zeros(this.n_repeat,this.getNGlm(),this.getNShadingCorrector());
            this.test_var_array = zeros(this.n_repeat,this.getNGlm(),this.getNShadingCorrector());
            
            this.training_bias2_array = zeros(this.n_repeat,this.getNGlm(),this.getNShadingCorrector());
            this.test_bias2_array = zeros(this.n_repeat,this.getNGlm(),this.getNShadingCorrector());
            
            this_glm_array(this.n_repeat,this.getNGlm(),this.getNShadingCorrector()) = MeanVar_GLM();
            this.glm_array = this_glm_array;
            
            this.training_index_array = cell(this.n_repeat,this.getNGlm(),this.getNShadingCorrector());
            this.test_index_array = cell(this.n_repeat,this.getNGlm(),this.getNShadingCorrector());
            
            this.saveSegmentation(scan.getSegmentation());
            this.rand_stream = rand_stream;
        end
        
        
        %DO EXPERIMENT
        function doExperiment(this)
            
            %do n_repeat times
            while (this.i_repeat <= this.n_repeat)
            
                %use its random stream
                RandStream.setGlobalStream(this.rand_stream);
                %for each glm
                for i_shad = 1:this.getNShadingCorrector()
                    %for each shading correction
                    for i_glm = 1:this.getNGlm()
                        %do one iteration of the experiment
                        this.doIteration(i_glm, i_shad);
                    end
                end
                
                %print the progress
                this.printProgress(this.i_repeat / this.n_repeat);
                %increment i_repeat
                this.i_repeat = this.i_repeat + 1;
                
            end
        end
        
        %DO ONE ITERATION OF EXPERIMENT
        function doIteration(this,i_glm,i_shad)
            %get the training and test mse
            [training_error, test_error] = this.trainingTestMeanVar(i_glm, i_shad);
            %save the training and test mse in the array
            this.training_msse_array(this.i_repeat,i_glm,i_shad) = training_error(1);
            this.training_mse_array(this.i_repeat,i_glm,i_shad) = training_error(2);
            this.test_msse_array(this.i_repeat,i_glm,i_shad) = test_error(1);
            this.test_mse_array(this.i_repeat,i_glm,i_shad) = test_error(2);
        end
        
        %TRAINING/TEST MEAN VAR
        %Gets the training and test MSE when fitting and predicting the mean and variance relationship
        %PARAMETERS:
            %glm_index: integer, pointing to which glm to use
            %shading_index: integer, pointing to which shading corrector to use
        %RETURN:
            %training_error: two vector [msse; mse] 
            %test_error: two vector [msse; mse]
        function [training_error, test_error, parameter] = trainingTestMeanVar(this, glm_index, shading_index)
            
            %get the model
            model = this.getGlm(this.shape_parameter, glm_index);

            %get random index of the training and test data
            index_suffle = randperm(this.n_sample);
            training_index = index_suffle(1:this.n_train);
            test_index = index_suffle((this.n_train+1):end);

            %get variance mean data of the training set
            [sample_mean,sample_var] = this.getMeanVar(training_index, shading_index);

            %train the classifier
            model.train(sample_mean,sample_var);
            %get the training mse
            training_error = model.getPredictionMSSE(sample_mean,sample_var);
            
            %save training_index, test_index and the trained model
            this.training_index_array{this.i_repeat, glm_index, shading_index} = training_index;
            this.test_index_array{this.i_repeat, glm_index, shading_index} = test_index;
            this.glm_array(this.i_repeat, glm_index, shading_index) = model;

            %get the variance mean data of the test set
            [sample_mean,sample_var] = this.getMeanVar(test_index, shading_index);

            %get the test mse
            test_error = model.getPredictionMSSE(sample_mean,sample_var);

            %get the glm parameter
            parameter = model.parameter;

        end
        
        function getVarBiasResult(this)
            this.i_repeat = 1;
            while (this.i_repeat <= this.n_repeat)
                
                for i_shad = 1:this.getNShadingCorrector()
                    
                    for i_glm = 1:this.getNGlm()
                        
                        training_index = this.training_index_array{this.i_repeat,i_glm,i_shad};
                        test_index = this.test_index_array{this.i_repeat,i_glm,i_shad};
                        
                        [var, bias2] = this.getVarBias(training_index, i_glm, i_shad);
                        this.training_var_array(this.i_repeat,i_glm,i_shad) = var;
                        this.training_bias2_array(this.i_repeat,i_glm,i_shad) = bias2;
                        
                        [var, bias2] = this.getVarBias(test_index, i_glm, i_shad);
                        this.test_var_array(this.i_repeat,i_glm,i_shad) = var;
                        this.test_bias2_array(this.i_repeat,i_glm,i_shad) = bias2;
                        
                    end
                    
                end
                
                %print the progress
                this.printProgress(this.i_repeat / this.n_repeat);
                %increment i_repeat
                this.i_repeat = this.i_repeat + 1;
                
            end
            
        end
        
        function [var, bias2] = getVarBias(this, index, i_glm, i_shad)
            [sample_mean,sample_var] = this.getMeanVar(index, i_shad);
            y_mean = zeros(numel(sample_mean),this.n_repeat);
            for j = 1:this.n_repeat
                y_mean(:,j) = this.glm_array(this.i_repeat,i_glm,i_shad).predict(sample_mean);
            end
            y_predict = y_mean(:,this.i_repeat);
            y_mean = mean(y_mean,2);
            var = mean((y_predict - y_mean).^2);
            bias2 = mean((y_mean - sample_var).^2);
        end
        
        %PRINT RESULTS
        %Box plot the training and test MSSE
        function printResults(this)
            
            %get array of shading corrector names
            shading_corrector_array = cell(this.getNShadingCorrector(),1);
            for i_shad = 1:this.getNShadingCorrector()
                [shading_corrector,reference_index] = this.getShadingCorrector(i_shad);
                shading_corrector.n_image = numel(reference_index);
                shading_corrector_array{i_shad} = shading_corrector.getName();
            end
            
            %get array of glm names
            glm_name_array = cell(this.getNGlm(),1);
            for i_glm = 1:this.getNGlm()
                model = this.getGlm([],i_glm);
                glm_name_array{i_glm} = model.getName();
            end
            
            %for the training error, then the test error
            for i = 1:2
                %produce figure
                figure;
                ax = gca;
                for i_shad = 1:this.getNShadingCorrector()
                    plot(0,0);
                    hold on;
                end
                %get the colours for each hold on
                colour_order = ax.ColorOrder;
                %for each shading correction
                for i_shad = 1:this.getNShadingCorrector()
                    %get the position of the box plot for this current shading correction
                    position = (1:this.getNGlm())-0.25+0.5*(i_shad)/(this.getNShadingCorrector()+1);
                    %get the array of errors
                    if i == 1
                        array = this.training_error_array(:,:,i_shad);
                    else
                        array = this.test_error_array(:,:,i_shad);
                    end
                    %box plot the errors
                    boxplot(array,'Position',position,'boxstyle','filled','medianstyle','target','outliersize',4,'symbol','o','Color',colour_order(i_shad,:));
                end
                %retick the x axis
                ax.XTick = 1:this.getNGlm();
                %label each glm with its name
                ax.XTickLabelRotation = 45;
                ax.XTickLabel = glm_name_array;
%                 if i == 1
%                     array = this.training_error_array();
%                 else
%                     array = this.test_error_array();
%                 end
%                 ax.YLim = [min(min(min(array))),max(max(max(array)))];
                %label the axis and legend
                if i == 1
                    ylabel('Training MSSE');
                else
                    ylabel('Test MSSE');
                end
                legend(shading_corrector_array);
            end 
        end
        
        %PLOT FULL FIT
        %Plot the variance and mean histogram, along with the fitted glm
        %Using all n_sample images, for all GLM
        function plotFullFit(this)

            %shape parameter is number of (images - 1)/2, this comes from the chi
            %squared distribution
            scan = this.getScan();
            full_shape_parameter = (scan.n_sample-1)/2;

            %for each shading corrector
            for i_shad = 1:this.getNShadingCorrector()
                %for each glm
                for i_glm = 1:this.getNGlm()
                    
                    %get the glm
                    model = this.getGlm(full_shape_parameter, i_glm);

                    %get the sample mean and variance
                    [sample_mean,sample_var] = this.getMeanVar(1:numel(this.greyvalue_array(1,:,1)),i_shad);
                    
                    %train the glm
                    model.train(sample_mean,sample_var);
                    
                    %indicate sample means and variances which are not outliers
                    mean_not_outlier = removeOutliers_iqr(sample_mean);
                    var_not_outlier = removeOutliers_iqr(sample_var);
                    %get boolean vector, true for values which are not outliers for both mean and variance
                    not_outlier = mean_not_outlier & var_not_outlier;
                    %remove outliers in the vector sample_mean and sample_var
                    sample_mean = sample_mean(not_outlier);
                    sample_var = sample_var(not_outlier);

                    %plot the frequency density
                    figure;
                    ax = hist3Heatmap(sample_mean,sample_var,[this.getNBin(),this.getNBin()],true);
                    colorbar;
                    hold on;

                    %get a range of greyvalues to plot the fit
                    x_plot = linspace(ax.XLim(1),ax.XLim(2),100);
                    %get the variance prediction along with the error bars
                    [variance_prediction, up_error, down_error] = model.predict(x_plot');

                    %plot the fit/prediction
                    plot(x_plot,variance_prediction,'r');
                    %plot the error bars
                    plot(x_plot,up_error,'r--');
                    plot(x_plot,down_error,'r--');
                    
                    xlabel('mean (arb. unit)');
                    ylabel('variance (arb. unit^2)');
                end
            end
            
            this.deleteVariables();
        end %plotFullFit
        
        %SAVE SEGMENTATION
        %Given segmentation from a scan object, save it as a vector
        function saveSegmentation(this, segmentation)
            this.segmentation = reshape(segmentation,[],1);
        end
        
        %GET MEAN VARIANCE
        %Get mean and variance vector using the images indicated by the parameter data_index
        %The mean and variance are already segmented
        %PARAMETERS:
            %image_index: vector of integers, points to which images to use for mean and variance estimation
            %shading_index: which shading correction to use, integer
        %RETURNS:
            %sample_mean: mean vector
            %sample_var: variance vector
        function [sample_mean,sample_var] = getMeanVar(this, image_index, shading_index)
            %get the scan object
            scan = this.getScan();
            %get the shading corrector
            [shading_corrector, reference_index] = this.getShadingCorrector(shading_index);
            %add the shading corrector
            scan.addShadingCorrector(shading_corrector, reference_index);
            
            %load the images and reshape it to be a design matrix
            image_stack = scan.loadImageStack(image_index);
            image_stack = reshape(image_stack,scan.area,numel(image_index));
            
            %segment the design matrix
            image_stack = image_stack(this.segmentation,:);
            
            %work out the mean and variance
            sample_mean = mean(image_stack,2);
            sample_var = var(image_stack,[],2);
        end
        
        %IMPLEMENTED: GET N BIN
        function n_bin = getNBin(this)
            n_bin = 100;
        end
        
    end
    
    methods (Abstract)
        
        %returns scan object
        scan = getScan(this);
        
        %returns number of glm to investigate
        n_glm = getNGlm(this);
        
        %returns glm model given index
        %index can range from 1 to getNGlm()
        model = getGlm(this, shape_parameter, index);
        
        %returns number of shading correctors to investigate
        n_shad = getNShadingCorrector(this);
        
        %returns shading corrector given index
        %index can range from 1 to getNShadingCorrector
        %RETURNS:
            %shading_corrector: ShadingCorrector object
            %reference_index: row vector containing integers
                %pointing to which reference scans to be used for shading correction training
        [shading_corrector, reference_index] = getShadingCorrector(this, index);
        
    end
    
end

