classdef Experiment_SimulateRoc < Experiment_NoDefect
    
    properties
    end
    
    methods (Access = public)
        
        %CONSTRUCTOR
        function this = Experiment_SimulateRoc(name)
            this@Experiment_NoDefect(name);
        end
        
        function printResults(this)
            
            for i_parameter = 1:numel(this.parameter_array)
                figure;
                ax = gca;
                for i_repeat = 1:this.n_repeat
                    index = ((i_repeat-1)*numel(this.size_array)+1) : (i_repeat*numel(this.size_array));
                    x = this.fdr_array(index, 1, i_parameter );
                    y = this.fdr_array(index, 2, i_parameter );
                    plot(x,y,'Color',ax.ColorOrder(1,:));
                    hold on;
                end
                scatter(this.fdr_array(:,1,i_parameter),this.fdr_array(:,2,i_parameter),'MarkerEdgeColor',ax.ColorOrder(2,:),'Marker','x');
            end
            
        end
        
    end
    
    %PROTECTED METHODS
    methods (Access = protected)
        
        %OVERRIDE: SETUP EXPERIMENT
        %PARAMETERS:
            %rng: random number generator
            %parameter_array: array of parameters for the function
        function setup(this, rng)
            this.rng = rng;
            this.parameter_array = 10.^linspace(2,3,20);
            this.size_array = linspace(0,5,41);
            this.n_repeat = 5;
            this.fdr_array = zeros(this.n_repeat*numel(this.size_array), 2, numel(this.parameter_array));
            this.plot_index = [17;10]; %size then parameter
            this.i_iteration = 0;
            this.n_iteration = numel(this.parameter_array) * this.n_repeat;
        end
        
        %METHOD: GET FDR
        %Given a convolution, do the test, get the false positive rate and save it
        %PARAMETERS:
            %convolution: EmpericalConvolution object
            %defect_simulator: the defect simulator for this current iteration
            %i_parameter: iteration integer
            %i_repeat: interation integer
            %i_size: iteration integer
        function getFdr(this, convolution, defect_simulator, i_parameter, i_repeat, i_size)
            %set the threshold of the test and do the test
            convolution.setSigma(this.size_array(i_size));
            convolution.doTest();
            %get the false positive rate
            this.fdr_array( (i_repeat-1)*numel(this.size_array)+i_size, 1, i_parameter ) = sum(sum(convolution.sig_image & (~defect_simulator.sig_image)))/defect_simulator.n_null;
            %get the true positive rate
            this.fdr_array( (i_repeat-1)*numel(this.size_array)+i_size, 2, i_parameter ) = sum(sum(convolution.sig_image & (defect_simulator.sig_image)))/defect_simulator.n_sig;
        end
        
        %METHOD: GET DATA
        %Return an object containing images
        function data = getData(this)
            data = AbsBlock_July16_30deg();
            data.addDefaultShadingCorrector();
        end
        
    end
    
end

