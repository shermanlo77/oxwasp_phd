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
        
        %temp variable, dim 1: for each pixel, dim 2: for each image
        greyvalue_array;
        segmentation;
        
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
            this.saveSegmentation();
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
            
            f = repmat(mean(this.y_array,2),1,this.n_bootstrap);
            noise_plot = var(this.y_array,[],2);
            for i_model = 1:this.getNModel()
                
                rss_plot = mean( (this.y_array - this.y_predict(:,:,i_model)).^2,2);
                mse_plot = mean( (this.y_predict(:,:,i_model) - f).^2,2);
                bias_plot = mean(this.y_predict(:,:,i_model),2) - mean(this.y_array,2);
                var_plot = var(this.y_predict(:,:,i_model),[],2);
                
                figure;
                plot(this.x_plot,rss_plot);
                hold on;
                plot(this.x_plot,noise_plot);
                legend('RSS','\sigma^2');
                xlabel('mean greyvalue (arb. unit)');
                ylabel('statistic (arb. unit^2)');

                figure;
                plot(this.x_plot,mse_plot);
                hold on;
                plot(this.x_plot,var_plot);
                plot(this.x_plot,bias_plot.^2);
                legend('MSE','VAR','BIAS^2');
                xlabel('mean greyvalue (arb. unit)');
                ylabel('statistic (arb. unit^2)');

                figure;
                plot(this.x_plot,bias_plot);
                xlabel('mean greyvalue (arb. unit)');
                ylabel('bias (arb. unit)');
               
            end
            
            
        end
        
        %SAVE SEGMENTATION
        %Given segmentation from a scan object, save it as a vector
        function saveSegmentation(this)
            scan = this.getScan();
            this.segmentation = scan.getSegmentation();
            this.segmentation = reshape(this.segmentation,[],1);
        end
        
        %GET MEAN VARIANCE
        %Get mean and variance vector using the images indicated by the parameter data_index
        %The mean and variance are already segmented
        %PARAMETERS:
            %image_index: vector of integers, points to which images to use for mean and variance estimation
        %RETURNS:
            %sample_mean: mean vector
            %sample_var: variance vector
        function [sample_mean,sample_var] = getMeanVar(this, image_index)
            %work out the mean and variance
            sample_mean = mean(this.greyvalue_array(:,image_index),2);
            sample_var = var(this.greyvalue_array(:,image_index),[],2);
        end
        
        %SAVE GREY VALUE ARRAY
        %Set up the member variable greyvalue_array
        function saveGreyvalueArray(this)
            
            %get the number of segmented pixels
            n_pixel = sum(this.segmentation);
            
            %get the scan object
            scan = this.getScan();
            
            %declare the array greyvalue array
            this.greyvalue_array = zeros(n_pixel, scan.n_sample); 
            
            %load the images and reshape it to be a design matrix
            image_stack = scan.loadImageStack();
            image_stack = reshape(image_stack,scan.area,scan.n_sample);

            %segment the design matrix
            this.greyvalue_array = image_stack(this.segmentation,:);
            
            this.n_train = round(scan.n_sample/2);
            this.y_array = zeros(this.n_plot, this.n_bootstrap);
            this.y_predict = zeros(this.n_plot, this.n_bootstrap, this.getNModel());
            this.shape_parameter = (this.n_train-1)/2;
            this.n_image = scan.n_sample;
            
            [sample_mean,~] = getMeanVar(this, 1:this.n_image);
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

