classdef (Abstract) Experiment_GLMVarMean_July16_30deg  < Experiment_GLMVarMean_July16
    
    properties
        i_shading_correction; %integer between 1 and number of shading correctors to investigate
        %subclasses can use this member variable to set a random stream, independent from each shading corrector
    end
    
    methods
        
        %CONSTRUCTOR
        function this = Experiment_GLMVarMean_July16_30deg(experiment_name)
            %call superclass with experiment name
            this@Experiment_GLMVarMean_July16(experiment_name);
        end
        
        %OVERRIDE: SET UP EXPERIMENT
        function setUpExperiment(this, rand_stream)
            %call superclass with 100 repeats and a random stream
            this.setUpExperiment@Experiment_GLMVarMean_July16(rand_stream);
        end
        
        function scan = getScan(this)
            scan = AbsBlock_July16_30deg();
        end

        function n_shad = getNShadingCorrector(this)
            n_shad = 1;
        end
    end
    
end

