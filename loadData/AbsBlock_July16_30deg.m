classdef AbsBlock_July16_30deg < AbsBlock_July16
    
    properties
    end
    
    methods
        
        function this = AbsBlock_July16_30deg()
            this@AbsBlock_July16('data/absBlock_noFilter_July16/scans/phantom_30deg/', 'block30deg_');
            this.addARTistFile('data/absBlock_noFilter_July16/sim/phantom/sim_block30.tif');
        end
        
    end
    
end

