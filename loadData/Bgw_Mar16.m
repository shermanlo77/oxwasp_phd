classdef Bgw_Mar16 < Scan
    
    properties
    end
    
    methods
        
        function this = Bgw_Mar16()
            this@Scan([], [], 1996, 1996, 100);
            %assign member variables
            this.panel_counter = PanelCounter_Brass();
            
            reference_scan_array(3) = Scan();
            this.reference_scan_array = reference_scan_array;
            this.reference_scan_array(1) = Scan('data/bgw_Mar16/black/', 'black_140316_', this.width, this.height, 20);
            this.reference_scan_array(2) = Scan('data/bgw_Mar16/grey/', 'grey_140316_', this.width, this.height, 20);
            this.reference_scan_array(3) = Scan('data/bgw_Mar16/white/', 'white_140316_', this.width, this.height, 20);
        end
    end
    
end

