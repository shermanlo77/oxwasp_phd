classdef Experiment_VarBias2 < Experiment
    
    %MEMBER VARIABLES
    properties
        
        rand_stream;
        
        n_bootstrap; %number of bootstrap samples
        shape_parameter; %shape_parameter of the gamma glm
        
        %row vector of greyvalue mean
        x_plot;
        n_plot;
        
        %matrix of test set variances
            %dim 1: for each x in x_plot
            %dim 2: for each fold
        y_array;
        
        %matrix of predicted variances
            %dim 1: for each x in x_plot
            %dim 2: for each fold
            %dim 3: for each model
        y_predict;
        
        %temporary variable
        %stores the grey values of each masked pixel, for each image
        mean_variance_estimator;
        
        n_train;
        n_image;

    end
    
    methods
        
        function this = Experiment_VarBias2()
            this@Experiment('VarBias2');
        end
        
        function setUpExperiment(this)
            this.n_bootstrap = 1000;
            this.n_plot = 100;
            this.rand_stream = RandStream('mt19937ar','Seed',uint32(3653123410));
        end
        
        function doExperiment(this)
            this.saveGreyvalueArray();
            RandStream.setGlobalStream(this.rand_stream);
            for i_bootstrap = 1:this.n_bootstrap
                index = randperm(this.n_image);
                train_index = index(1:this.n_train);
                test_index = index((this.n_train+1):end);
                
                [sample_mean,sample_var] = getMeanVar(this, train_index);
                
                for i_model = 1:this.getNModel()
                   model = this.getModel(i_model);
                   model.train(sample_mean, sample_var);
                   this.y_predict(:,i_bootstrap,i_model) = model.predict(this.x_plot);                   
                end
                
                [sample_mean,sample_var] = getMeanVar(this, test_index);
                for i_x = 1:this.n_plot
                    [~,var_index] = min(abs(sample_mean - this.x_plot(i_x)));
                    this.y_array(i_x,i_bootstrap) = sample_var(var_index);
                end
                
                this.printProgress(i_bootstrap/this.n_bootstrap);
                
            end
        end
        
        %DELETE GREYVALUE ARRAY
        function deleteVariables(this)
            this.greyvalue_array = [];
        end
        
        function printResults(this)
            
            %get array of glm names
            model_name_array = cell(this.getNModel(),1);
            for i = 1:this.getNModel()
                model = this.getModel(i);
                model_name_array{i} = model.getFileName();
            end
            
            f = repmat(mean(this.y_array,2),1,this.n_bootstrap);
            noise_plot = var(this.y_array,[],2);
            for i_model = 1:this.getNModel()
                
                rss_plot = mean( (this.y_array - this.y_predict(:,:,i_model)).^2,2);
                mse_plot = mean( (this.y_predict(:,:,i_model) - f).^2,2);
                bias_plot = mean(this.y_predict(:,:,i_model),2) - mean(this.y_array,2);
                var_plot = var(this.y_predict(:,:,i_model),[],2);
                
                fig = figure_latexSub();
                plot(this.x_plot,rss_plot);
                hold on;
                plot(this.x_plot,noise_plot);
                legend('RSS','\sigma^2','Location','northwest');
                xlabel('mean greyvalue (arb. unit)');
                ylabel('statistic (arb. unit^2)');
                saveas(fig,fullfile('reports','figures',strcat('noise',model_name_array{i_model},'.eps')),'epsc');

                fig = figure_latexSub();
                plot(this.x_plot,mse_plot);
                hold on;
                plot(this.x_plot,var_plot);
                plot(this.x_plot,bias_plot.^2);
                legend('MSE','VAR','BIAS^2','Location','northwest');
                xlabel('mean greyvalue (arb. unit)');
                ylabel('statistic (arb. unit^2)');
                saveas(fig,fullfile('reports','figures',strcat('mse',model_name_array{i_model},'.eps')),'epsc');

                fig = figure_latexSub();
                plot(this.x_plot,bias_plot);
                xlabel('mean greyvalue (arb. unit)');
                ylabel('bias (arb. unit)');
                saveas(fig,fullfile('reports','figures',strcat('bias',model_name_array{i_model},'.eps')),'epsc');
               
            end
            
            
        end
        
        %GET MEAN VARIANCE
        %Get mean and variance vector using the images indicated by the parameter image_index
        %PARAMETERS:
            %image_index: vector of integers, points to which images to use for mean and variance estimation
        %RETURNS:
            %sample_mean: mean vector
            %sample_var: variance vector
        function [sample_mean,sample_var] = getMeanVar(this, image_index)
            %work out the mean and variance
            [sample_mean,sample_var] = this.mean_variance_estimator.getMeanVar(image_index);
        end
        
        %SAVE GREY VALUE ARRAY
        %Set up the member variable greyvalue_array
        function saveGreyvalueArray(this)
            
            scan = this.getScan();
            this.mean_variance_estimator = MeanVarianceEstimator(scan);
            this.mean_variance_estimator.saveGreyvalueArray(scan);
            
            this.n_train = round(scan.n_sample/2);
            this.y_array = zeros(this.n_plot, this.n_bootstrap);
            this.y_predict = zeros(this.n_plot, this.n_bootstrap, this.getNModel());
            this.shape_parameter = (this.n_train-1)/2;
            this.n_image = scan.n_sample;
            
            [sample_mean,~] = this.getMeanVar(1:this.n_image);
            this.x_plot = (linspace(min(sample_mean),max(sample_mean),this.n_plot))';
                
        end %saveGreyvalueArray
        
        function scan = getScan(this)
           scan = AbsBlock_Sep16_30deg();
           scan.addDefaultShadingCorrector();
        end
        
        function n_model = getNModel(this)
            n_model = 6;
        end
        
        %IMPLEMENTED: GET GLM
        function model = getModel(this, index)
            switch index
                case 1
                    model = MeanVar_GLM(this.shape_parameter,1,LinkFunction_Identity());
                case 2
                    model = MeanVar_GLM(this.shape_parameter,-1,LinkFunction_Canonical());
                case 3
                    model = MeanVar_GLM(this.shape_parameter,-2,LinkFunction_Canonical());
                case 4
                    model = MeanVar_GLM(this.shape_parameter,1,LinkFunction_Log());
                case 5
                    model = MeanVar_GLM(this.shape_parameter,-1,LinkFunction_Log());
                case 6
                    model = MeanVar_kNN(1E3);
            end
        end
        
    end
    
end

