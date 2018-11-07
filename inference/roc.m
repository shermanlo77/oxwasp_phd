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
  %nPoints: number of significant levels to use to get the ROC, eg 41
%RETURN:
  %falsePositive: column vector of false positive rates
  %truePositive: column vector of the corresponding true positive rates
  %areaRoc: area under the ROC curve, done using trapeziums
function [falsePositive, truePositive, areaRoc] = roc(zImage, altImage, nPoints)

  %declare array to store the rates
  falsePositive = zeros(nPoints, 1);
  truePositive = zeros(nPoints, 1);
  
  n = numel(altImage); %number of pixels
  nAlt = sum(sum(altImage)); %number of alt pixels
  nNull = n - nAlt; %number of null pixels
  
  alphaArray = linspace(0, 1, nPoints); %array of significant levels to try out
  
  zTester = ZTester(zImage); %instantiate z tester for testing using BH procedure
  
  %for each significant level
  for i = 1:nPoints
    %set the significant level and do the hypothesis test
    zTester.setSize(alphaArray(i));
    zTester.doTest();
    %get the false and ture positive rate
    falsePositive(i) = sum(sum(zTester.sig_image & (~altImage))) / nNull;
    truePositive(i) = sum(sum(zTester.sig_image & (altImage))) / nAlt;
  end
  
  %get area of roc curve using trapeziums
  areaRoc = sum( 0.5 * (truePositive(1:(end-1))+truePositive(2:end)) ...
      .*(falsePositive(2:end)-falsePositive(1:(end-1))) );

end

