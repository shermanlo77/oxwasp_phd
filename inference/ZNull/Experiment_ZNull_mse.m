classdef Experiment_ZNull_mse < Experiment
    
    properties (SetAccess = protected)
        rng;
        n_bootstrap;
        k_plot;
        k_optima;
        k_optima_bootstrap;
        error_regress; %dim 1: for each k, dim 2: for each n
        lambda_array;
        glm_array;
        plot_index;
        i_iteration;
        n_iteration;
    end
    
    methods (Access = public)
        
        %CONSTRUCTOR
        function this = Experiment_ZNull_mse()
            this@Experiment('ZNull_mse');
        end
        
        function printResults(this)
            previous = this.getPreviousResult();
            for i_n = 1:numel(previous.n_array)
    
                array = this.getObjective(previous.var_array(:,:,i_n));
                figure;
                box_plot = Boxplots(array,true);
                box_plot.setPosition(previous.k_array);
                box_plot.plot();
                hold on;
                plot(this.k_plot,this.error_regress(:,i_n));
                ylabel('ln mean squared error');
                xlabel('kernel width');
                title(strcat('n=',num2str(previous.n_array(i_n))));

            end
            

            boxplot_k_optima = Boxplots(this.k_optima_bootstrap,true);
            boxplot_k_optima.setPosition((previous.n_array).^(-1/5));
            figure;
            boxplot_k_optima.plot();
            hold on;
            
            x_plot = linspace(previous.n_array(1).^(-1/5),previous.n_array(end).^(-1/5),100);
            [y_plot, up_error, down_error] = this.glm_array{this.plot_index}.predict(x_plot);
            ax = plot(x_plot,y_plot);
            plot(x_plot, up_error, 'Color', ax.Color,'LineStyle',':');
            plot(x_plot, down_error, 'Color', ax.Color,'LineStyle',':');
            xlabel('n^{-1/5}');
            ylabel('optimal kernel width');
            
            coeff_array = zeros(2,numel(this.lambda_array));
            error_array = zeros(2,numel(this.lambda_array));
            for i_lambda = 1:numel(this.lambda_array)
                [beta,beta_err] = this.glm_array{i_lambda}.getParameter();
                coeff_array(:,i_lambda) = beta;
                error_array(:,i_lambda) = beta_err;
            end
            figure;
            errorbar(this.lambda_array,coeff_array(1,:),error_array(1,:));
            hold on;
            errorbar(this.lambda_array,coeff_array(2,:),error_array(2,:));
            xlabel('smoothness');
            ylabel('estimated parameter');
            legend('intercept','gradient');
        end

    end
    
    methods (Access = protected)
        
        %IMPLEMENTED: SET UP EXPERIMENT
        function setup(this)
            
            %get the results from the ZNull experiment
            previous = this.getPreviousResult();
            %if the experiment is not completed, return error
            if ~previous.is_complete
                error('Experiment_ZNull not completed');
            end
            
            %assign member variables
            this.rng = RandStream('mt19937ar','Seed',uint32(3027942399));
            this.n_bootstrap = 100;
            this.k_plot = interp1(1:numel(previous.k_array),previous.k_array,linspace(1,numel(previous.k_array),10*numel(previous.k_array)));
            this.k_optima = zeros(numel(previous.n_array),1);
            this.k_optima_bootstrap = zeros(this.n_bootstrap,numel(previous.n_array));
            this.error_regress = zeros(numel(this.k_plot),numel(previous.n_array));
            this.plot_index = 4;
            this.lambda_array = linspace(0.02,0.2,10);
            this.glm_array = cell(numel(this.lambda_array),1);
            
            this.i_iteration = 0;
            this.n_iteration = this.n_bootstrap * numel(previous.n_array) + numel(this.lambda_array);
            
        end
        
        function doExperiment(this)
            
            previous = this.getPreviousResult();
            x_glm = previous.n_array.^(-1/5);
            
            for i_lambda = 1:numel(this.lambda_array)
                
                lambda = this.lambda_array(i_lambda);
                
                k_optima_i = zeros(numel(previous.n_array),1);
                error_regress_i = zeros(numel(this.k_plot),numel(previous.n_array));
            
                for i_n = 1:numel(previous.n_array)

                    ln_mse = this.getObjective(previous.var_array(:,:,i_n));
                    x =  repmat(previous.k_array,previous.n_repeat,1);
                    y =  reshape(ln_mse',[],1);
                    x(isnan(y)) = [];
                    y(isnan(y)) = [];
                    n = numel(x);

                    fitter = this.getRegression(lambda);
                    fitter.train(x,y);
                    error_regress_i(:,i_n) = fitter.predict(this.k_plot);
                    k_optima_i(i_n) = this.findOptima(error_regress_i(:,i_n));

                    if i_lambda == this.plot_index
                        for i_bootstrap = 1:this.n_bootstrap
                            bootstrap_index = this.rng.randi(n,n,1);
                            fitter = this.getRegression(lambda);
                            fitter.train(x(bootstrap_index),y(bootstrap_index));
                            error_bootstrap = fitter.predict(this.k_plot);
                            this.k_optima_bootstrap(i_bootstrap, i_n) = this.findOptima(error_bootstrap);

                            this.i_iteration = this.i_iteration + 1;
                            this.printProgress(this.i_iteration / this.n_iteration);
                        end
                    end
                end
                
                if i_lambda == this.plot_index
                    this.k_optima = k_optima_i;
                    this.error_regress = error_regress_i;
                end

                glm_gamma = GlmGamma(1,IdentityLink());
                glm_gamma.train(x_glm,k_optima_i);
                glm_gamma.estimateShapeParameter();
                this.glm_array{i_lambda} = glm_gamma;
                
                this.i_iteration = this.i_iteration + 1;
                this.printProgress(this.i_iteration / this.n_iteration);
                
            end
            
            
        end
        
        function fitter = getRegression(this, lambda)
            fitter = LocalQuadraticRegression(GaussianKernel(),lambda);
            %fitter = LocalQuadraticRegression(EpanechnikovKernel(),0.3);
        end
        
        function previous = getPreviousResult(this)
            previous = Experiment_ZNull();
        end
        
        function objective = getObjective(this, var_array)
            objective = log((var_array.^2-1).^2);
        end
        
        function optima = findOptima(this,regress)
            [~,i_k_optima] = min(regress);
            optima = this.k_plot(i_k_optima);
        end
       
        
    end
    
end

