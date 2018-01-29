classdef Experiment_ZNull_mse < Experiment
    
    properties (SetAccess = protected)
        rng;
        n_bootstrap;
        k_plot;
        k_optima;
        k_optima_bootstrap;
        error_regress; %dim 1: for each k, dim 2: for each n
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
            
            y = this.k_optima;
            x = previous.n_array.^(-1/5);
            glm_gamma = GlmGamma(1,IdentityLink());
            glm_gamma.train(x,y);
            glm_gamma.estimateShapeParameter();

            boxplot_k_optima = Boxplots(this.k_optima_bootstrap,true);
            boxplot_k_optima.setPosition((previous.n_array).^(-1/5));
            figure;
            boxplot_k_optima.plot();
            hold on;
            
            x_plot = linspace(previous.n_array(1).^(-1/5),previous.n_array(end).^(-1/5),100);
            [y_plot, up_error, down_error] = glm_gamma.predict(x_plot);
            ax = plot(x_plot,y_plot);
            plot(x_plot, up_error, 'Color', ax.Color,'LineStyle',':');
            plot(x_plot, down_error, 'Color', ax.Color,'LineStyle',':');
            xlabel('n^{-1/5}');
            ylabel('optimal kernel width');
            
            [a,b] = glm_gamma.getParameter()
            
%             y_scale = std(y);
%             y = y/y_scale;
%             X = [ones(numel(previous.n_array),1),previous.n_array.^(-1/5)];
%             x_shift = mean(X(:,2));
%             x_scale = std(X(:,2));
%             X(:,2) = (X(:,2)-x_shift)/x_scale;
%             [~,~,stats] = glmfit(X,y,'gamma','link','identity','constant','off');
%             y_scale * stats.beta(2)/x_scale
%             sqrt(stats.covb(end))*y_scale/x_scale
%             y_scale * (stats.beta(1) - stats.beta(2)*x_shift/x_scale)
%             sqrt(y_scale^2*stats.covb(1)+(x_shift*y_scale/x_scale)^2*stats.covb(end) + 2*y_scale*(x_shift*y_scale/x_scale)*stats.covb(2))
           
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
            
            this.i_iteration = 0;
            this.n_iteration = this.n_bootstrap * numel(previous.n_array);
            
        end
        
        function doExperiment(this)
            
            previous = this.getPreviousResult();
            
            for i_n = 1:numel(previous.n_array)
                
                ln_mse = this.getObjective(previous.var_array(:,:,i_n));
                x =  repmat(previous.k_array,previous.n_repeat,1);
                y =  reshape(ln_mse',[],1);
                x(isnan(y)) = [];
                y(isnan(y)) = [];
                n = numel(x);

                fitter = this.getRegression();
                fitter.train(x,y);
                this.error_regress(:,i_n) = fitter.predict(this.k_plot);
                
                this.k_optima(i_n) = this.findOptima(this.error_regress(:,i_n));
                
                for i_bootstrap = 1:this.n_bootstrap
                    bootstrap_index = this.rng.randi(n,n,1);
                    fitter = this.getRegression();
                    fitter.train(x(bootstrap_index),y(bootstrap_index));
                    error_bootstrap = fitter.predict(this.k_plot);
                    this.k_optima_bootstrap(i_bootstrap, i_n) = this.findOptima(error_bootstrap);
                    
                    this.i_iteration = this.i_iteration + 1;
                    this.printProgress(this.i_iteration / this.n_iteration);
                end
                
            end
            
            
        end
        
        function fitter = getRegression(this)
            fitter = LocalQuadraticRegression(GaussianKernel(),0.1);
            %fitter = LocalLinearRegression(EpanechnikovKernel(),0.5);
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

