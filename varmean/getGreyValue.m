%MIT License
%Copyright (c) 2019 Sherman Lo

%FUNCTION: GET GRAY VALUE
%Returns an array of grey values from the ROI of the scan
function greyValueArray = getGreyValue(scan)
  %get the segmentation of the scan
  %reshape the segmentation to be a column vector
  segmentation = reshape(scan.getSegmentation(),[],1);
  %load the images and reshape it to be a design matrix
  imageStack = scan.loadImageStack();
  imageStack = reshape(imageStack,scan.area,scan.nSample);
  %segment the design matrix and save it to greyvalue_array
  greyValueArray = imageStack(segmentation,:);
end

