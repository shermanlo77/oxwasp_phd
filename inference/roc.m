%MIT License
%Copyright (c) 2019 Sherman Lo

%FUNCTION: ROC (Receiver operating characteristic)
%Given an image of z statistics and a map of where the null and alt pixels are, return points for
%the roc curve.
%
%The roc curve is a plot of false positive rate vs true positive rate, this can be obtained by
%varying the significant level. The area under the ROC can be used as an indiciator of the
%performance of the classifier
%
%PARAMETERS:
  %zImage: image of z statistics
  %altImage: binary image, true if that pixel is alt, else null
%RETURN:
  %falsePositive: column vector of false positive rates
  %truePositive: column vector of the corresponding true positive rates
  %areaRoc: area under the ROC curve
function [falsePositive, truePositive, areaRoc] = roc(zImage, altImage)

  n = numel(altImage); %number of pixels
  nAlt = sum(sum(altImage)); %number of alt pixels
  nNull = n - nAlt; %number of null pixels
  nPoints = n+1; %include alpha level 0

  %declare array to store the rates
  falsePositive = zeros(nPoints, 1);
  truePositive = zeros(nPoints, 1);
  
  pImage = 2*(1-normcdf(abs(zImage)));
  pSorted = sort(reshape(pImage, [], 1));
  pSorted = [0; pSorted];
  
  %for each significant level
  for i = 1:numel(pSorted)
    p = pSorted(i);
    positiveImage = pImage <= p;
    %get the false and true positive rate
    falsePositive(i) = sum(sum(positiveImage & (~altImage))) / nNull;
    truePositive(i) = sum(sum(positiveImage & (altImage))) / nAlt;
  end
  
  %get area of roc curve
  areaRoc = zeros(nPoints-1, 1);
  for i = 1:(nPoints-1)
    areaRoc(i) = truePositive(i) * (falsePositive(i+1) - falsePositive(i));
  end
  areaRoc = sum(areaRoc);

end

