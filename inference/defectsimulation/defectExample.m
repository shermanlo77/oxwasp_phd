%PROCEDURE: DEFECT EXAMPLE
%Produce an image with a defect and contamination, use the empirical null filter to recover the
    %image from the contamination. The hypothesis test (or defect detectiong) are compared before
    %and after contamination.
%Plot the defected contaminated z image
%Plot the empirical null mean image
%Plot the empirical null std image
%Plot the ROC before and after contamination
%For the z_alpha=2 level, print:
  %type 1 error
  %type 2 error
  %area under ROC
%PARAMETERS:
  %defectSimulator: simulate a defected and contaminated image
  %imageSize: size of the image
  %radius: radius of the kernel
function defectExample(defectSimulator, imageSize, radius)

  %get the defected image
  [image, isAltImage, imagePreBias] = defectSimulator.getDefectedImage([imageSize, imageSize]);

  filter = EmpiricalNullFilter(radius); %filter it
  filter.setNInitial(3);
  filter.filter(image);

  %get the empirical null and the filtered image
  imageFiltered = filter.getFilteredImage();
  nullMean = filter.getNullMean();
  nullStd = filter.getNullStd();

  %get the image pre/post bias with significant pixels highlighted
  zTesterPreBias = ZTester(imagePreBias);
  zTesterPreBias.doTest();
  zTesterPostBias = ZTester(imageFiltered);
  zTesterPostBias.doTest();
  
  %plot the contaminated z image
  figure;
  zmin = min(min(image));
  zmax = max(max(image));
  imagePlot = ImagescSignificant(image);
  imagePlot.setCLim([zmin,zmax]);
  imagePlot.plot();
  
  %empirical null plot
  figure;
  imagesc(nullMean);
  colorbar;
  figure;
  imagesc(nullStd);
  colorbar;

  %work out true and false positive before and after bias adding
  %print out the area under the roc
  [falsePositivePreBias, truePositivePreBias, areaRocPreBias] = roc(imagePreBias, isAltImage, 100);
  [falsePositivePostBias, truePositivePostBias, areaRocPostBias] = ...
      roc(imageFiltered, isAltImage, 100);
  figure;
  plot(falsePositivePreBias, truePositivePreBias);
  hold on;
  plot(falsePositivePostBias, truePositivePostBias);
  plot([0,1],[0,1],'k--');
  xlabel('false positive rate');
  ylabel('true positive rate');
  legend('pre bias adding','post bias adding','Location','southeast');

  %for this particular significant level, print the type 1 and type 2 error
  %type 1 = false positive
  %type 2 = false negative
  disp('Pre contamination');
  type1Error = sum(sum(zTesterPreBias.sig_image(~isAltImage))) / sum(sum(~isAltImage));
  type2Error = sum(sum(~(zTesterPreBias.sig_image(isAltImage)))) / sum(sum(isAltImage));
  disp('type 1 error');
  disp(type1Error);
  disp('type 2 error');
  disp(type2Error);
  disp('roc area');
  disp(areaRocPreBias);

  disp('For post bias adding');
  type1Error = sum(sum(zTesterPostBias.sig_image(~isAltImage))) / sum(sum(~isAltImage));
  type2Error = sum(sum(~(zTesterPostBias.sig_image(isAltImage)))) / sum(sum(isAltImage));
  disp('type 1 error');
  disp(type1Error);
  disp('type 2 error');
  disp(type2Error);
  disp('roc area');
  disp(areaRocPostBias);

end

