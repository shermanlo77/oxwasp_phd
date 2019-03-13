%FUNCTION: INFERENCE EXAMPLE
%Sets up the zImage for the AbsBlock_Sep16_120deg() dataset
%19 images are used to train the variance-mean relationship
%1 (held out) image is then compared with the aRTist simulation, obtaining a z image
function [test, artist, zImage] = inferenceExample()

  %set random seed
  randStream = RandStream('mt19937ar','Seed',uint32(3538096789));

  %load data and add shading correction
  blockData = AbsBlock_Sep16_120deg();
  blockData.addDefaultShadingCorrector();

  %get random permutation for each image
  index = randStream.randperm(blockData.n_sample);
  nTest = 1;
  nTrain = blockData.n_sample - nTest;
  trainingIndex = index(1:nTrain);
  testIndex = index((nTrain+1):(nTrain+nTest));

  %get a phanton image and aRTist image
  artist = blockData.getShadingCorrectedARTistImage(ShadingCorrector(),1:blockData.reference_white);

  %get the segmentation image
  segmentation = blockData.getSegmentation();

  %get the training images
  trainingStack = blockData.loadImageStack(trainingIndex);
  %segment the image
  trainingStack = reshape(trainingStack,blockData.area,nTrain);
  trainingStack = trainingStack(reshape(segmentation,[],1),:);
  %get the segmented mean and variance greyvalue
  training_mean = mean(trainingStack,2);
  training_var = var(trainingStack,[],2);

  %train glm using the training set mean and variance
  model = GlmGamma(1,IdentityLink());
  model.setShapeParameter((nTrain-1)/2);
  model.train(training_mean,training_var);

  %predict variance given aRTist
  varPredict = reshape(model.predict(reshape(artist,[],1)),blockData.height, blockData.width);

  %get the test images
  test = blockData.loadImageStack(testIndex);

  %get the z statistic
  zImage = (test - artist)./sqrt(varPredict);
  %set non segmented pixels to be nan
  zImage(~segmentation) = nan;
  
end