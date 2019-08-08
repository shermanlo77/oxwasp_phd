%FUNCTION: GET Z IMAGE
%Returns the z image given a Scan object
%
%Adds linear shading correction
%19 images are used to train the variance-mean relationship
%1 (held out) image is then compared with the aRTist simulation, obtaining a z image
%
%PARAMETERS:
  %scan: Scan object containing the projection to do inference on
  %rng: random number generator
%RETURN:
  %zImage: image containing z statistics
  %test: the test projection, the one which gets compared with aRTist
  %artist: the aRTist projection
function [zImage, test, artist] = getZImage(scan, rng)

  %add shading correction
  scan.addShadingCorrectorLinear();

  %get random permutation for each image
  index = rng.randperm(scan.nSample);
  nTest = 1;
  nTrain = scan.nSample - nTest;
  trainingIndex = index(1:nTrain);
  testIndex = index((nTrain+1):(nTrain+nTest));

  %get a phanton image and aRTist image
  artist = scan.getArtistImageShadingCorrected('ShadingCorrector', 1:scan.whiteIndex);

  %get the segmentation image
  segmentation = scan.getSegmentation();

  %get the training images
  trainingStack = scan.loadImageStack(trainingIndex);
  %segment the image
  trainingStack = reshape(trainingStack,scan.area,nTrain);
  trainingStack = trainingStack(reshape(segmentation,[],1),:);
  %get the segmented mean and variance greyvalue
  X = mean(trainingStack,2);
  XMean = mean(X);
  XStd = std(X);
  X = (X-XMean)/XStd;
  Y = var(trainingStack,[],2);
  YStd = std(Y);
  Y = Y/YStd;
  
  %train glm using the training set mean and variance
  model = fitglm(X, Y, 'Distribution', 'gamma', 'Link', 'identity');

  %predict variance given aRTist
  XArtist = reshape(artist,[],1);
  XArtist = (XArtist - XMean) / XStd;
  varPredict = reshape(model.predict(XArtist),scan.height, scan.width) * YStd;

  %get the test images
  test = scan.loadImageStack(testIndex);

  %get the z statistic
  zImage = (test - artist)./sqrt(varPredict);
  %set non segmented pixels to be nan
  zImage(~segmentation) = nan;
  
end