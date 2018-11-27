%ABSTRACT CLASS: EXPERIMENT SUB ROI DEFECT DETECT
%Inference is done in separate segmentated section of the image
classdef (Abstract) ExperimentSubRoiDefectDetect < ExperimentDefectDetect
  
  properties
  end
  
  methods (Access = public)
    
    %CONSTRUCTOR
    function this = ExperimentSubRoiDefectDetect(experimentName)
      this@ExperimentDefectDetect(experimentName);
    end
    
  end
  
  methods (Access = protected)
    
    %OVERRIDE: DO EXPERIMENT
    %For each radius, filter the zImage with a filter with that radius
    function doExperiment(this)
      
      %for each radius in radius Array
      for iRadius = 1:numel(this.radiusArray)
        
        %for each segmentation
        for iSegmentation = 1:this.scan.nSubSegmentation
        
          %filter the image
          radius = this.radiusArray(iRadius);
          filter = EmpiricalNullFilter(radius);
          filter.filterRoi(this.zImage, this.scan.getSubRoiPath(iSegmentation));
          
          %get the resulting output images so far          
          zFilterImage = this.zFilterArray(:,:,iRadius);
          nullMeanImage = this.nullMeanArray(:,:,iRadius);
          nullStdImage = this.nullStdArray(:,:,iRadius);
          
          %get the resulting segmentated output images so far
          zFilterSegmentation = filter.getFilteredImage();
          nullMeanSegmentation = filter.getNullMean();
          nullStdSegmentation = filter.getNullStd();
          
          %put the segmentated output images into the resulting output image
          segmentation = this.scan.getSubSegmentation(iSegmentation);
          zFilterImage(segmentation) = zFilterSegmentation(segmentation);
          nullMeanImage(segmentation) = nullMeanSegmentation(segmentation);
          nullStdImage(segmentation) = nullStdSegmentation(segmentation);
          
          %save the resulting output image
          this.zFilterArray(:,:,iRadius) = zFilterImage;
          this.nullMeanArray(:,:,iRadius) = nullMeanImage;
          this.nullStdArray(:,:,iRadius) = nullStdImage;
        
        end
        
        %print progress
        this.printProgress(iRadius/numel(this.radiusArray));
        
      end
      
    end
    
  end
  
end

