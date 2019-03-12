%CLASS: Z TESTER
%Does hypothesis test on an image of z statistics
%Multiple testing corrected using FDR, see:
  %Benjamini, Y. and Hochberg, Y., 1995.
  %Controlling the false discovery rate: a practical and powerful approach to multiple testing.
  %Journal of the royal statistical society. Series B (Methodological), pp.289-300.
%The null hypothesis can be corrected using the empirical null
%How to use:
  %pass z statistics into the constructor (can be 2d array)
  %optional: estimate the null hypothesis, call the method estimateNull(initialValue, seed)
  %call the method doTest();
  %p values are in the member variable pImage
  %positive pixels are in the member variable positiveImage
classdef ZTester < handle

  %MEMBER VARIABLES
  properties (SetAccess = protected)
    zImage; %2d array of z statistics
    nullMean = 0; %mean of the null hypothesis
    nullStd = 1; %std of the null hypothesis
    threshold = 0.05; %fdr of the test
    sizeCorrected; %corrected size of the test due to multiple testing

    pImage; %2d array of p_values
    positiveImage; %boolean 2d array, true if that pixel is significant
    nTest; %number of tests
  end

  %METHODS
  methods (Access = public)

    %CONSTRUCTOR
    %PARAMETERS:
      %zImage: 2d array of z statistics
    function this = ZTester(zImage)
      %assign member variables
      this.zImage = zImage;
      %get the number of non_nan values in zImage
      this.nTest = sum(sum(~isnan(zImage)));
    end

    %METHOD: SET THRESHOLD
    %Set the fdr of the hypothesis test
    %PARAMETERS:
      %threshold: the fdr threshold
    function setThreshold(this, threshold)
      this.threshold = threshold;
    end

    %METHOD: ESTIMATE NULL
    %Estimates the nullMean and nullStd
    %PARAMETERS:
      %initialValue: value to start the newton raphson from
      %seed: int32 for setting the random number generator, used for using different initial values
    function estimateNull(this, initialValue, seed)
      empiricalNull = EmpiricalNull(reshape(this.zImage,[],1), initialValue, seed);
      empiricalNull.estimateNull();
      this.nullMean = empiricalNull.getNullMean();
      this.nullStd = empiricalNull.getNullStd();
    end

    %METHOD: GET Z CORRECTED
    %Returns the z image, corrected under the empirical null hypothesis
    %RETURNS:
      %zCorrected: 2d array of z statistics, corrected under the empirical null hypothesis
    function zCorrected = getZCorrected(this)
      zCorrected = (this.zImage - this.nullMean) ./ (this.nullStd);
    end

    %METHOD: DO TEST
    %Does hypothesis using the p values, corrected using FDR
    %Saves positive pixels in the member variable positiveImage
    function doTest(this)
      %calculate the p values
      this.pImage = 2*(normcdf(-abs(this.getZCorrected())));
      %instantise p tester and do test
      pTester = PTester(this.pImage, this.threshold);
      pTester.doTest();
      %save the results, positiveImage and sizeCorrected
      this.positiveImage = pTester.positiveImage; %2d boolean of significant pixels
      this.sizeCorrected = pTester.size; %size of test, corrected for multiple testing
    end

    %METHOD: GET Z CRITICAL
    %Return the critical values of z, corrected using the emperical null
    function zCritical = getZCritical(this)
      zCritical = norminv(this.sizeCorrected/2);
      zCritical(2) = -zCritical;
      zCritical = zCritical*this.nullStd + this.nullMean;
    end
    
    %METHOD: PLOT HISTOGRAM
    %Plot histogram of z statistics
    function plotHistogram(this)
      histogramCustom(reshape(this.zImage(),[],1));
    end

    %METHOD: PLOT HISTOGRAM WITH NULL DISTRIBUTION AND CRITICAL BOUNDARY
    %Produce a figure plots:
      %histogram of z statistics
      %empirical null (optional)
      %critical boundary
    %PARAMETERS:
      %wantNull: plot the empirical null as well?
    function plotHistogram2(this, wantNull)
      %plot histogram
      this.plotHistogram();
      hold on;
      ax = gca;
      if (wantNull)
        %get values from min to max
        zPlot = linspace(ax.XLim(1), ax.XLim(2), 500);
        %plot null
        this.plotNull(zPlot);
      end
      %plot critical boundary
      this.plotCritical();
      
      %swap the order of the critical and histogram
      ax.Children = flip(ax.Children);
      
      %label axis and legend
      xlabel('z statistic');
      ylabel('frequency density');
      if (wantNull)
        legend(ax.Children(1:(end-1)), 'z statistic', 'null', 'critical');
      else
        legend(ax.Children(1:(end-1)), 'z statistic', 'critical');
      end
    end

    %METHOD: PLOT P VALUES
    %Plot the p values in order with the BH critical region
    function plotPValues(this)
      this.plotPValues2(true(size(this.zImage)));
      ax = gca;
      legend(ax.Children([1,3]), 'p value', 'critical', 'Location','northwest');
    end

    %METHOD: PLOT P VALUES (highlight null and non-null)
    %Plot the p values in order with the BH critical region
    %The null and non-null p values have different symbols
    %PARAMETERS:
      %isNull: boolean image, true if this statistic is null
    function plotPValues2(this, isNull)
      this.plotOrderedPValues(isNull);
      ax = gca;
      legend(ax.Children, 'null','alt','critical','Location','northwest');
    end
    
  end
  
  methods (Access = private)
    
    %METHOD: PLOT NULL DENSITY
    %PARAMETERS:
      %zPlot: values to evalute the null density at
    function plotNull(this, zPlot)
      plot(zPlot, this.nTest * normpdf(zPlot, this.nullMean, this.nullStd));
    end
    
    %METHOD: PLOT CRITICAL BOUNDARY
    %Plot a vertical area to show critical boundary on a histogram
    function plotCritical(this)
      ax = gca;
      critical = this.getZCritical();
      xlimBefore = ax.XLim;
      %left hand side
      this.plotArea([xlimBefore(1),critical(1)], [ax.YLim(2),ax.YLim(2)]);
      hold on;
      %right hand size
      this.plotArea([critical(2),xlimBefore(2)], [ax.YLim(2),ax.YLim(2)]);
      ax.XLim = xlimBefore;
    end
    
    %METHOD: PLOT BH CRITICAL
    %Plot the BH critical region on a p value plot
    function plotBhCritical(this)
      %plot the BH critical line
      ax = gca;
      this.plotArea(orderIndex, this.sizeCorrected/this.nTest * 1:this.nTest);
      %set the scale to log
      ax.XScale = 'log';
      ax.YScale = 'log';
      %set other graph properties
      ax.XLim = [1,this.nTest];
    end
    
    %METHOD: PLOT ORDERED P VALUES
    %Scatter plot the p values vs order
    %Use different symbol for null and non-null
    %PARAMETERS:
      %isNull: boolean image, true if this statistic is null
    function plotOrderedPValues(this, isNull)
      
      %remove any nan and convert to vector
      isNull(isnan(this.pImage)) = [];
      pVector = this.pImage(~isnan(this.pImage));
      
      %get array of x axis values : integers representing the order
      orderIndex = 1:this.nTest;
      %scatter plot the ordered p values
      [pVector, index] = sort(pVector);
      %reorder the boolean vector isNull because of sort
      isNull = isNull(index);
      
      %plot the BH critical line
      this.plotArea(orderIndex, this.threshold/this.nTest*orderIndex);
      hold on;
      %plot the p values
      scatter(orderIndex(~isNull), pVector(~isNull),'rx');
      scatter(orderIndex(isNull), pVector(isNull),'b.');
      xlabel('order');
      ylabel('p value');
      
      %set the scale to log and other graph properties
      ax = gca;
      ax.XScale = 'log';
      ax.YScale = 'log';
      ax.XLim = [1,this.nTest];
      
    end

    %METHOD: PLOT AREA
    %Plot area to show critical region
    function plotArea(this, x, y)
      axArea = area(x,y);
      axArea.LineStyle = 'none';
      axArea.FaceColor = [0.9686, 0.8666, 0.8196];
      ax = gca;
      ax.Layer = 'top'; %make the minor tick show up
    end
    
  end

end

