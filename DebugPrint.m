classdef DebugPrint
  
  methods (Static)
    
    function newFile(prefix)
      debug = uk.ac.warwick.sip.empiricalnullfilter.DebugPrint();
      debug.newFile(prefix);
    end
    
    function write(string)
      debug = uk.ac.warwick.sip.empiricalnullfilter.DebugPrint();
      debug.write(string);
    end
    
    function close()
      debug = uk.ac.warwick.sip.empiricalnullfilter.DebugPrint();
      debug.close();
    end
    
  end
  
end

