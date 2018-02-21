%CLASS: EXPERIMENT MEAN VARIANCE FIT
%Fits models onto the mean and variance data
classdef Experiment_MeanVarFit < Experiment_GlmMse

    %MEMBER VARIABLES
    properties (SetAccess = protected)
        
        %array of within pixel mean and variance
            %dim 1: for each pixel
            %dim 2: for each shading correction
        mean_array;
        var_array;
        
        %array of means to evaluate the variance prediction
            %dim 1: n_linspace elements
            %dim 2: for each shading correction
        x_array
        
        %array of variance prediction, upper and lower bound error for each value in x
            %dim 1: n_linspace elements
            %dim 2: for each model
            %dim 3: for each shading correction
        predict_array;
        up_array;
        down_array;
        
        %number of points to evaluate the prediction
        n_linspace;
        
    end
    
    %METHODS
    methods (Access = public)
        
        %CONSTRUCTOR
        %PARAMETERS:
            %name: nameof the experiment
        function this = Experiment_MeanVarFit(name)
            this@Experiment_GlmMse(name);
        end
        
        %OVERRIDE: PRINT RESULTS
        function printResults(this)
            %get histogram obect
            hist_plot = Hist3Heatmap();
            
            %for each shading correction
            for j_shad = 1:this.getNShadingCorrector()
                %for each glm
                for j_glm = 1:this.getNGlm()

                    %plot the frequency density
                    fig = LatexFigure.sub();
                    hist_plot.plot(this.mean_array(:,j_shad),this.var_array(:,j_shad));
                    hold on;

                    %plot the fit/prediction
                    plot(this.x_array(:,j_shad),this.predict_array(:,j_glm,j_shad),'r');
                    %plot the error bars
                    plot(this.x_array(:,j_shad),this.up_array(:,j_glm,j_shad),'r--');
                    plot(this.x_array(:,j_shad),this.down_array(:,j_glm,j_shad),'r--');
                    %label the axis
                    xlabel('mean (arb. unit)');
                    ylabel('variance (arb. unit)');
                    
                    %save the figure
                    saveas(fig,fullfile('reports','figures','meanVar',strcat(this.experiment_name,'_iShad',num2str(j_shad),'_iGlm',num2str(j_glm),'.eps')),'epsc');
                end
            end
        end
        
    end
    
    methods (Access = protected)
        
        %OVERRIDE: SETUP
        function setup(this)
            %call superclass
            %1 repeat, no rand_stream
            this.setup@Experiment_GlmMse(1,[]);
        end
        
        %OVERRIDE: ASSIGN ARRAY
        %Assign member variables
        function assignArray(this)
            %get the number of pixels in the image
            scan = this.getScan();
            n_pixel = sum(sum(scan.getSegmentation()));
            %assign member variables
            this.n_linspace = 500;
            this.mean_array = zeros(n_pixel,this.getNShadingCorrector());
            this.var_array = zeros(n_pixel,this.getNShadingCorrector());
            this.x_array = zeros(this.n_linspace,this.getNShadingCorrector());
            this.predict_array = zeros(this.n_linspace,this.getNGlm(),this.getNShadingCorrector());
            this.up_array = zeros(this.n_linspace,this.getNGlm(),this.getNShadingCorrector());
            this.down_array = zeros(this.n_linspace,this.getNGlm(),this.getNShadingCorrector());
        end
        
        %OVERRIDE: SAVE GREY VALUE ARRAY
        %Calls superclass version
        %Also saves the within pixel mean and variance in the member variables mean_array and var_array
        function saveGreyvalueArray(this)
            %call superclass version
            this.saveGreyvalueArray@Experiment_GlmMse();
            %get the within pixel mean and variance using all n_sample images
            [sample_mean,sample_var] = this.getMeanVar(1:this.n_sample);
            %evaluate the prediction over a range between the min and max
            x_plot = linspace(min(sample_mean),max(sample_mean),this.n_linspace);
            %save the x_plot values, mean and variance to the member variable
            this.x_array(:,this.i_shad) = x_plot';
            this.mean_array(:,this.i_shad) = sample_mean;
            this.var_array(:,this.i_shad) = sample_var;
        end
        
        %OVERRIDE: DO ITERATION
        %Train the model and save it's prediction
        function doIteration(this)
            
            %get the glm
            model = this.getGlm(this.i_glm);

            %train the glm
            model.train(this.mean_array(:,this.i_shad),this.var_array(:,this.i_shad));

            %get the variance prediction along with the error bars
            if model.hasErrorbar()
                [variance_prediction, up_error, down_error] = model.predict(this.x_array(:,this.i_shad));
            else
                variance_prediction = model.predict(this.x_array(:,this.i_shad)');
                up_error = nan;
                down_error = nan;
            end
            
            %save the prediction and error bars
            this.predict_array(:,this.i_glm,this.i_shad) = variance_prediction;
            this.up_array(:,this.i_glm,this.i_shad) = up_error;
            this.down_array(:,this.i_glm,this.i_shad) = down_error;
        end
        
    end
    
end

