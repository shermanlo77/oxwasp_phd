classdef Experiment_ImageVarBias2 < Experiment_GLMVarMean
    
    properties
        
        y_array;
        y_hat_array;
        rss_array;
        var_array;
        sigma_array;
        
    end
    
    methods
        
        function this = Experiment_ImageVarBias2(experiment_name)
            this@Experiment_GLMVarMean(experiment_name);
        end
        
        function setUpExperiment(this,n_repeat, rand_stream)
             this.setUpExperiment@Experiment_GLMVarMean(n_repeat, rand_stream);
        end
        
        %OVERRIDE
        function assignArray(this)
            n_pixel = sum(this.mean_variance_estimator.segmentation);
            this.y_array = zeros(n_pixel, this.getNGlm(), this.getNShadingCorrector());
            this.y_hat_array = zeros(n_pixel, this.getNGlm(), this.getNShadingCorrector());
            this.rss_array = zeros(n_pixel, this.getNGlm(), this.getNShadingCorrector());
            this.var_array = zeros(n_pixel, this.getNGlm(), this.getNShadingCorrector());
            this.sigma_array = zeros(n_pixel, this.getNGlm(), this.getNShadingCorrector());
        end
        
        %DO ONE ITERATION OF EXPERIMENT
        function doIteration(this)
            %get the model
            model = this.getGlm(this.i_glm);

            %get random index of the training and test data
            index_suffle = randperm(this.n_sample);
            training_index = index_suffle(1:this.n_train);
            test_index = index_suffle((this.n_train+1):end);

            %get variance mean data of the training set
            [sample_mean,sample_var] = this.getMeanVar(training_index);
            %train the classifier
            model.train(sample_mean,sample_var);
            
            %get the variance mean data of the test set
            [~,y] = this.getMeanVar(test_index);
            y_hat = model.predict(sample_mean);
            
            this.y_array(:,this.i_glm, this.i_shad) = this.meanUpdate(this.y_array(:,this.i_glm, this.i_shad), y);
            this.y_hat_array(:,this.i_glm, this.i_shad) = this.meanUpdate(this.y_hat_array(:,this.i_glm, this.i_shad), y_hat);
            this.rss_array(:,this.i_glm, this.i_shad) = this.meanUpdate(this.rss_array(:,this.i_glm, this.i_shad), (y_hat - y).^2);
            
            this.var_array(:,this.i_glm, this.i_shad) = this.varianceUpdate(this.var_array(:,this.i_glm, this.i_shad), this.y_hat_array(:,this.i_glm, this.i_shad), y_hat);
            this.sigma_array(:,this.i_glm, this.i_shad) = this.varianceUpdate(this.sigma_array(:,this.i_glm, this.i_shad), this.y_array(:,this.i_glm, this.i_shad), y);
            
        end
        
        %PRINT RESULTS
        %Box plot the training and test MSSE
        function printResults(this)
            this.plotBoxPlot(this.rss_array,'RSS');
            this.plotBoxPlot(this.var_array,'Variance');
            this.plotBoxPlot(this.y_hat_array - this.y_array,'bias');
            this.plotBoxPlot((this.y_hat_array - this.y_array).^2,'bias^2');
            this.plotBoxPlot(this.var_array + (this.y_hat_array - this.y_array).^2,'MSE');
        end
        
        
        function new_statistic = meanUpdate(this, old_statistic, new_value)
            if this.i_repeat == 1
                new_statistic = new_value;
            else
                new_statistic = ((this.i_repeat - 1)/this.i_repeat)*old_statistic + new_value/this.i_repeat;
            end
        end
        
        function new_statistic = varianceUpdate(this, old_statistic, new_mean, new_value)
           if this.i_repeat == 1
               new_statistic = new_value;
           elseif this.i_repeat == 2
               new_statistic = ((old_statistic - new_mean).^2 + (new_value - new_mean).^2)/2;
           else
               new_statistic = ((this.i_repeat - 1)/this.i_repeat)*old_statistic + ((new_value - new_mean).^2)/(this.i_repeat - 1);
           end
        end

    end
    
    methods (Abstract)
        
        %returns scan object
        scan = getScan(this);
        
        %returns number of glm to investigate
        n_glm = getNGlm(this);
        
        %returns glm model given index
        %index can range from 1 to getNGlm()
        model = getGlm(this, index);
        
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
