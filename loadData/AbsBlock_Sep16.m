classdef AbsBlock_Sep16 < Scan
    
    properties
    end
    
    methods
        
        function this = AbsBlock_Sep16(folder_location, file_name)
            this@Scan(folder_location, file_name, 2000, 2000, 20, 80, 20, 500);
            
            reference_scan_array(5) = Scan();
            this.reference_scan_array = reference_scan_array;
            this.reference_scan_array(1) = Scan('data/absBlock_CuFilter_Sep16/scans/blank_0W/', '0w_', this.width, this.height, 20, 0, 0, this.time_exposure);
            this.reference_scan_array(2) = Scan('data/absBlock_CuFilter_Sep16/scans/blank_5W/', '5w_', this.width, this.height, 20, this.voltage, 4.96, this.time_exposure);
            this.reference_scan_array(3) = Scan('data/absBlock_CuFilter_Sep16/scans/blank_10W/', '10w_', this.width, this.height, 20, this.voltage, 10, this.time_exposure);
            this.reference_scan_array(4) = Scan('data/absBlock_CuFilter_Sep16/scans/blank_15W/', '15w_', this.width, this.height, 20, this.voltage, 15.04, this.time_exposure);
            this.reference_scan_array(5) = Scan('data/absBlock_CuFilter_Sep16/scans/blank_20W/', '20w_', this.width, this.height, 20, this.voltage, 20, this.time_exposure);

            this.reference_scan_array(2).addARTistFile('data/absBlock_CuFilter_Sep16/sim/blank/a5.tif');
            this.reference_scan_array(3).addARTistFile('data/absBlock_CuFilter_Sep16/sim/blank/a10.tif');
            this.reference_scan_array(4).addARTistFile('data/absBlock_CuFilter_Sep16/sim/blank/a15.tif');
            this.reference_scan_array(5).addARTistFile('data/absBlock_CuFilter_Sep16/sim/blank/a20.tif');
            
            this.reference_white = 5;
        end
        
    end
    
end

