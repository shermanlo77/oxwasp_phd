classdef Experiment_GlmDeviance_July16_30deg  < Experiment_GlmDeviance_July16
    
    properties
    end
    
    methods (Access = public)
        
        %CONSTRUCTOR
        function this = Experiment_GlmDeviance_July16_30deg()
            %call superclass with experiment name
            this@Experiment_GlmDeviance_July16('GlmDeviance_July16_30deg');
        end
        
    end
    
    methods (Access = protected)
        
        %OVERRIDE: SET UP EXPERIMENT
        function setup(this)
            %call superclass with 100 repeats and a random stream
            this.setup@Experiment_GlmDeviance_July16(RandStream('mt19937ar','Seed',uint32(3285278378)));
        end
        
        %IMPLEMENTED: GET SCAN
        function scan = getScan(this)
            scan = AbsBlock_July16_30deg();
        end

    end
    
end

