classdef AbsBlock_July16_30deg < Scan
    
    properties
    end
    
    methods
        
        function this = AbsBlock_July16_30deg()
            this@Scan('data/absBlock_noFilter_July16/scans/phantom_30deg/', 'block30deg_', 2000, 2000, 100, 80, 36, 708);
            
            this.addARTistFile('data/absBlock_noFilter_July16/sim/phantom/sim_block30.tif');
            
            [this.reference_scan_array, this.reference_white] = ReferenceArrayGetter.getReferenceScanArray_July16();
        end
        
        function addDefaultShadingCorrector(this)
            this.addShadingCorrector(ShadingCorrector(),[1,5]);
        end
        
    end
    
end

