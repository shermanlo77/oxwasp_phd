%CLASS: EXPERIMENT Z NULL MSE
%Experiment for investigating the performance of the emperical null var
%ln mean squared error vs kernel width are plotted and fitted using local quadratic regression
%the optimal kernel width is found using the minimum of the fitted curve
%optimal kernel width vs n^(-1/5) is plotted and straight line is plotted, boxplot is plotted by bootstraping the ln mean squared error vs kernel width data
%sensitivty analysis is done by varying the parameter of the local quadratic regression
%
%Plotted are:
%ln mean squared error vs kernel width for all n
%optimal kernel width vs n^(-1/5)
%gradient and intercept vs smoothness of local quadratic regression
classdef Experiment_ZNull_mse < Experiment
    
    %MEMBER VARIABLES
    properties (SetAccess = protected)
        rng; %random number generator
        n_bootstrap; %number of bootstrap for plotting purposes
        k_plot; %array of values of kernel widths to investigate
        k_optima; %optimal kernel width for each n
        
        %optimal kernel width for each n for each bootstrap
            %dim 1: for each bootstrap
            %dim 1: for each n
        k_optima_bootstrap;
        
        %the value of the local quadratic regression
            %dim 1: for each kernel width
            %dim 2: for each n
        error_regress;
        
        %array of local quadratic regression
        lambda_array;
        %array of glm fitter when fitting optimal kernel width vs n^(-1/5) for different local quadratic regressions
        glm_array;
        
        %the pointer of lambda to plot graphs in printResults
        plot_index;
        
        %progress bar variables
        i_iteration;
        n_iteration;
    end
    
    %METHODS
    methods (Access = public)
        
        %CONSTRUCTOR
        function this = Experiment_ZNull_mse()
            this@Experiment('ZNull_mse');
        end
        
        %IMPLEMENTED: RESULTS
        function printResults(this)
            %get the results from the previous experiment
            previous = this.getPreviousResult();
            %for each n
            for i_n = 1:numel(previous.n_array)
                %get the ln MSE
                array = this.getObjective(previous.correctionVarArray(:,:,i_n));
                fig = LatexFigure.sub();
                %boxplot the ln MSE for each kernel width
                box_plot = Boxplots(array,true);
                %set the values of the kernel width
                box_plot.setPosition(previous.k_array);
                box_plot.plot();
                hold on;
                %plot the local quadratic regression
                plot(this.k_plot,this.error_regress(:,i_n));
                %label axis and graph
                ylabel('ln squared error');
                xlabel('bandwidth');
                title(strcat('log n=',num2str(log10(previous.n_array(i_n)))));
                ylim([-15,5]);
                saveas(fig,fullfile('reports','figures','inference',strcat(this.experiment_name,'plot',num2str(i_n),'.eps')),'epsc');
            end
            
            %boxplot the optimal kernel widths
            boxplot_k_optima = Boxplots(this.k_optima_bootstrap,true);
            %vs n^(-1/5)
            boxplot_k_optima.setPosition((previous.n_array).^(-1/5));
            
            fig = LatexFigure.main();
            boxplot_k_optima.plot();
            hold on;
            %plot the glm fit of optimal kernel width vs n^(-1/5)
            x_plot = linspace(previous.n_array(1).^(-1/5),previous.n_array(end).^(-1/5),100);
            [y_plot, up_error, down_error] = this.glm_array{this.plot_index}.predict(x_plot);
            ax = plot(x_plot,y_plot);
            %plot the error bar of the glm fit
            plot(x_plot, up_error, 'Color', ax.Color,'LineStyle',':');
            plot(x_plot, down_error, 'Color', ax.Color,'LineStyle',':');
            %label axis
            xlabel('n^{-1/5}');
            ylabel('optimal bandwidth');
            saveas(fig,fullfile('reports','figures','inference',strcat(this.experiment_name,'rule_of_thumb','.eps')),'epsc');
            
            %declare array for storing the parameters
            coeff_array = zeros(2,numel(this.lambda_array));
            %declare array for storing the standard error of the parameters
            error_array = zeros(2,numel(this.lambda_array));
            %for each lambda (parameter of the local quadratic regression) or for each glm
            for i_lambda = 1:numel(this.lambda_array)
                %get the glm parameter and its standard error
                [beta,beta_err] = this.glm_array{i_lambda}.getParameter();
                %save the parameter and standard error to the array
                coeff_array(:,i_lambda) = beta;
                error_array(:,i_lambda) = beta_err;
            end
            
            %plot the glm coefficient vs lambda (parameter of the local quadratic regression)
            fig = LatexFigure.main();
            errorbar(this.lambda_array,coeff_array(1,:),error_array(1,:)); %plot intercept
            hold on;
            errorbar(this.lambda_array,coeff_array(2,:),error_array(2,:)); %plot gradient
            %label axis
            xlabel('lqr smoothness');
            ylabel('estimated parameter');
            legend('intercept','gradient','Location','east');
            saveas(fig,fullfile('reports','figures','inference',strcat(this.experiment_name,'sensitive','.eps')),'epsc');
            
            %save the intercept and gradient
            latex_table = LatexTable(coeff_array(:,this.plot_index), error_array(:,this.plot_index),{'Intercept','Gradient'} , {'Estimate'});
            latex_table.print(fullfile('reports','tables','ZNull_mse_glm_estimate.txt'));
            
            %save the bandwidth used in the plots
            file_id = fopen(fullfile('reports','tables','ZNull_bandwidth.txt'),'w');
            fprintf(file_id,'%.4f',this.lambda_array(this.plot_index));
            fclose(file_id);
        end

    end
    
    %PROTECTED METHODS
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
            this.k_plot = interp1(1:numel(previous.k_array),previous.k_array,linspace(1,numel(previous.k_array),100*numel(previous.k_array)));
            this.k_optima = zeros(numel(previous.n_array),1);
            this.k_optima_bootstrap = zeros(this.n_bootstrap,numel(previous.n_array));
            this.error_regress = zeros(numel(this.k_plot),numel(previous.n_array));
            this.plot_index = 6;
            this.lambda_array = linspace(0.02,0.3,20);
            this.glm_array = cell(numel(this.lambda_array),1);
            
            this.i_iteration = 0;
            this.n_iteration = numel(previous.n_array) * (this.n_bootstrap + numel(this.lambda_array));
            
        end
        
        %IMPLEMENTED: DO EXPERIMENT
        function doExperiment(this)
            %get the results of the previous experiment
            previous = this.getPreviousResult();
            %get array of values of n^(-1/5)
            x_glm = previous.n_array.^(-1/5);
            
            %for each lambda (parameter of the local quadratic regression)
            for i_lambda = 1:numel(this.lambda_array)
                %get the lambda
                lambda = this.lambda_array(i_lambda);
                %declare array of optimal kernel widths, one element for each n
                k_optima_i = zeros(numel(previous.n_array),1);
                %declare array of values of the local quadratic regression
                    %dim 1: for each kernel width
                    %dim 2: for each n
                error_regress_i = zeros(numel(this.k_plot),numel(previous.n_array));
            
                %for each n
                for i_n = 1:numel(previous.n_array)
                    %get the ln mse
                    ln_mse = this.getObjective(previous.correctionVarArray(:,:,i_n));
                    %declare array of kernel widths, taking into account the n_repeat in the previous experiment
                    x = repmat(previous.k_array,previous.n_repeat,1);
                    %declare array of ln_mse
                    y = reshape(ln_mse',[],1);
                    %remove any NAN
                    x(isnan(y)) = [];
                    y(isnan(y)) = [];
                    %get the number of data points
                    n = numel(x);

                    %fit y vs x (ln mse vs kernel width) using the regression
                    fitter = this.getRegression(lambda);
                    fitter.train(x,y);
                    %get the fitted regression for each kernel width
                    error_regress_i(:,i_n) = fitter.predict(this.k_plot);
                    %find the optimal kernel width
                    k_optima_i(i_n) = this.findOptima(error_regress_i(:,i_n));
                    
                    %update the progress bar
                    this.i_iteration = this.i_iteration + 1;
                    this.printProgress(this.i_iteration / this.n_iteration);

                    %if i_lambda is pointing to the lambda to be plotted
                    if i_lambda == this.plot_index
                        
                        %for this.n_boostrap times
                        for i_bootstrap = 1:this.n_bootstrap
                            %get the index of bootstrap samples
                            bootstrap_index = this.rng.randi(n,n,1);
                            %fit the regression on the boostrap data
                            fitter = this.getRegression(lambda);
                            fitter.train(x(bootstrap_index),y(bootstrap_index));
                            %get the value of the regression
                            error_bootstrap = fitter.predict(this.k_plot);
                            %find the optimal kernel width for this bootstrap
                            this.k_optima_bootstrap(i_bootstrap, i_n) = this.findOptima(error_bootstrap);
                            %update the progress bar
                            this.i_iteration = this.i_iteration + 1;
                            this.printProgress(this.i_iteration / this.n_iteration);
                        end
                    end
                end
                
                %if i_lambda is pointing to the lambda to be plotted
                if i_lambda == this.plot_index
                    %save the array of optimal kernel widths
                    this.k_optima = k_optima_i;
                    %save the evaluations of the local quadratic regression
                    this.error_regress = error_regress_i;
                end

                %instantise a Gamma glm for fitting optimal kernel width vs n^(-1/5)
                glm_gamma = GlmGamma(1,IdentityLink());
                %train the glm
                glm_gamma.train(x_glm,k_optima_i);
                glm_gamma.estimateShapeParameter();
                %save the glm
                this.glm_array{i_lambda} = glm_gamma;
   
            end
            
        end
        
        %METHOD: GET REGRESSION
        %PARAMETERS:
            %lambda: parameter of the local quadratic regression
        %RETURN:
            %fitter: local quadratic regression object
        function fitter = getRegression(this, lambda)
            fitter = LocalQuadraticRegression(GaussianKernel(),lambda);
        end
        
        %METHOD: GET PREVIOUS
        %RETURN:
            %previous: Experiment object from Z Null experiment
        function previous = getPreviousResult(this)
            previous = Experiment_ZNull();
        end
        
        %METHOD: GET OBJECTIVE
        %Return the ln mse
        %PARAMETERS:
            %var_array: array of estimated variances
        %RETURN:
            %objective: array of ln mse of the estimate variances
        function objective = getObjective(this, var_array)
            objective = log((sqrt(var_array)-1).^2);
        end
        
        %METHOD: FIND OPTIMA
        %PARAMETER:
            %regress: array of evaluations of the regression
        %RETURN:
            %optima: value of the index which minimises the regression
        function optima = findOptima(this,regress)
            [~,i_k_optima] = min(regress);
            optima = this.k_plot(i_k_optima);
        end
       
    end
    
end

