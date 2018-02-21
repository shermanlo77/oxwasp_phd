classdef Experiment_MeanVarFit_July16_30deg  < Experiment_MeanVarFit_July16
    
    properties
    end
    
    methods (Access = public)
        
        %CONSTRUCTOR
        function this = Experiment_MeanVarFit_July16_30deg()
            %call superclass with experiment name
            this@Experiment_MeanVarFit_July16('MeanVarFit_July16_30deg');
        end
        
    end
    
    methods (Access = protected)
        
        %IMPLEMENTED: GET SCAN
        function scan = getScan(this)
            scan = AbsBlock_July16_30deg();
        end

    end
    
end

