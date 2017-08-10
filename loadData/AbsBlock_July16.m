classdef AbsBlock_July16 < Scan
    
    properties
    end
    
    methods
        
        function this = AbsBlock_July16(folder_location, file_name)
            this@Scan(folder_location, file_name, 2000, 2000, 100, 80, 36, 708);
            
            reference_scan_array(6) = Scan();
            this.reference_scan_array = reference_scan_array;
            this.reference_scan_array(1) = Scan('data/absBlock_noFilter_July16/scans/blank_0W/', 'block_black_', this.width, this.height, 20, 0, 0, this.time_exposure);
            this.reference_scan_array(2) = Scan('data/absBlock_noFilter_July16/scans/blank_10W/', 'block10w_', this.width, this.height, 20, this.voltage, 10.08, this.time_exposure);
            this.reference_scan_array(3) = Scan('data/absBlock_noFilter_July16/scans/blank_18W/', 'block18w_', this.width, this.height, 20, this.voltage, 18.08, this.time_exposure);
            this.reference_scan_array(4) = Scan('data/absBlock_noFilter_July16/scans/blank_28W/', 'block28w_', this.width, this.height, 20, this.voltage, 28, this.time_exposure);
            this.reference_scan_array(5) = Scan('data/absBlock_noFilter_July16/scans/blank_36W/', 'block36w_', this.width, this.height, 20, this.voltage, 36, this.time_exposure);
            this.reference_scan_array(6) = Scan('data/absBlock_noFilter_July16/scans/blank_44W/', 'block44w_', this.width, this.height, 20, this.voltage, 44, this.time_exposure);

            this.reference_scan_array(2).addARTistFile('data/absBlock_noFilter_July16/sim/blank/a10.tif');
            this.reference_scan_array(3).addARTistFile('data/absBlock_noFilter_July16/sim/blank/a18.tif');
            this.reference_scan_array(4).addARTistFile('data/absBlock_noFilter_July16/sim/blank/a28.tif');
            this.reference_scan_array(5).addARTistFile('data/absBlock_noFilter_July16/sim/blank/a36.tif');
            this.reference_scan_array(6).addARTistFile('data/absBlock_noFilter_July16/sim/blank/a44.tif');
            
            this.reference_white = 5;
        end
        
    end
    
end

