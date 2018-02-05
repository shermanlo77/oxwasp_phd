classdef Experiment_SimulateRoc < Experiment_NoDefect
    
    properties
    end
    
    methods (Access = public)
        
        %CONSTRUCTOR
        function this = Experiment_SimulateRoc(name)
            this@Experiment_NoDefect(name);
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
            %defect_simulator: not used here
            %i_parameter: iteration integer
            %i_repeat: interation integer
            %i_size: iteration integer
            %n_pixel: number of pixels in the masked image
        function getFdr(this, convolution, defect_simulator, i_parameter, i_repeat, i_size, n_pixel)
            %set the threshold of the test and do the test
            convolution.setSigma(this.size_array(i_size));
            convolution.doTest();
            %get the false positive rate
            this.fdr_array( (i_repeat-1)*numel(this.size_array)+i_size, 1, i_parameter ) = sum(sum(convolution.sig_image & (~defect_simulator.sig_image)))/n_pixel;
            %get the true positive rate
            this.fdr_array( (i_repeat-1)*numel(this.size_array)+i_size, 2, i_parameter ) = sum(sum(convolution.sig_image & (defect_simulator.sig_image)))/n_pixel;
        end
        
        %METHOD: GET DATA
        %Return an object containing images
        function data = getData(this)
            data = AbsBlock_July16_30deg();
            data.addDefaultShadingCorrector();
        end
        
    end
    
end

