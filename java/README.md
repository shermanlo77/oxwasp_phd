# Mode Filter and Empirical Null Filter

* MIT License - all source code
* Copyright (c) 2020 Sherman Lo

*ImageJ* plugins for the mode filter and empirical null filter. The mode filter is an edge preserving smoothing filter by taking the mode of the empirical density. This may have applications in image processing such as image segmentation. The filters were also implemented on a GPU using *CUDA* and *JCuda*. This speeds up the filtering by a huge margin.

Where appropriate, please cite the thesis Lo, S.E. (2020). *Characterisation of Computed Tomography Noise in Projection Space with Applications to Additive Manufacturing*. PhD thesis, University of Warwick, Department of Statistics.

<img src=../publicImages/mandrillExample.jpg width=800><br>
The mode filter applied on the [Mandrill test image](http://sipi.usc.edu/database/database.php?volume=misc). Top left to top right, bottom left to bottom right: mandrill test image, then the mode filter with radius of 2, 4, 8, 16, 32, 64, 128 applied.

## How to Compile (Linux recommended)
Requires *Maven* as well as *Java Runtime Environment* and *Java Development Kit*. For the use of GPU, requires *CUDA Developement Kit* which should include a *nvcc* compiler.

Clone this repository.

Optional for GPU: go to `/java/cuda/` and run `make` to compile the *CUDA* code into a `.ptx` file.

Go to `/java/` and run

```
mvn package
```
to compile the *Java* code. The compiled `.jar` file is stored in `java/target/Empirical_Null_Filter-2.1.jar` and can be used as an *ImageJ* plugin. Copies of libraries are stored in `java/target/libs/` and would need to be installed in *ImageJ* as well.

## How to Install (*Fiji* recommended)
Installation can be done by copying the `.jar` files to *Fiji*'s directories. Or on *Fiji*, click on the *Plugins* menu followed by *Install...* (or Ctrl + Shift + M). Install by copying:
* `java/target/Empirical_Null_Filter-2.1.jar` to *Fiji*'s `/plugins/` directory
* `java/target/libs/commons-math3-3.6.1.jar` to *Fiji*'s `/jars/` directory. This may have been done for you already
* `java/target/libs/jcuda-10.1.0.jar` to *Fiji*'s `/jars/` directory
* `java/target/libs/jcuda-natives-10.1.0-linux-x86_64.jar` (or some variation) to *Fiji*'s `/jars/` directory

## Options
<img src=../publicImages/filter_gui.png><br>
<ul>
  <li>
    Number of initial values
    <ul>
      <li>
        Number of initial values for the Newton-Raphson method. Increase this for a more accurate filtering at a price of more computational time. Compared to other options, this has a big effort on the resulting image. The default value is 3 but should be in the order of 50-100 if this filter is to be applied to (non-Gaussian) images.
      </li>
    </ul>
  </li>
  <li>
    Number of steps
    <ul>
      <li>
        Number of iterations in the Newton-Raphson method. Increase this for a more accurate filtering at a price of more computational time.
      </li>
    </ul>
  </li>
  <li>
    Log tolerance (CPU version only)
    <ul>
      <li>
        The tolerance allowed for the Newton-Raphson method to accept the solution. Decrease this for a more accurate filtering at a price of more computational time.
      </li>
    </ul>
  </li>
  <li>
    Block dim x and y (GPU version only)
    <ul>
      <li>
        Sets the dimensions of the block of threads on the GPU. This affects the performance of the filter. Good suggestions are 16 and 32. Solutions are shared between neighbours within blocks.
      </li>
    </ul>
  </li>

</ul>

## About the Mode Filter
The mode filter is an image filter much like the mean filter and median filter. They process each pixel in an image. For a given pixel, the value of the pixel is replaced by the mean or median over all pixels within a distance *r* away. The mean and median filter can be used in *ImageJ*, it results in a smoothing of the image.

<img src=.././publicImages/filters.jpg width=800><br>
Top left: [Mandrill test image](http://sipi.usc.edu/database/database.php?volume=misc). Top right: Mean filter with radius 32. Bottom left: Median filter with radius 32. Bottom right: Mode filter with radius 32.

The mode filter is a by-product of the empirical null filter. Instead of taking the mean or median, the mode is taken, more specifically, the argmax of the empirical density. The optimisation problem was solved using the Newton-Raphson method. Various random initial values were tried to home in on the global maximum. Because the filtered image is expected to be smooth, the different initial values were influenced by neighbouring pixels to aid in the optimisation problem.

The resulting mode filtered image gives a smoothed image which has an impasto effect and preserved edges. This may have applications in noise removal or image segmentation.

The mode filtered was implemented on the CPU by modifying existing *Java* code from *ImageJ*. Each thread filters a row of the image in parallel from left to right. The solution to one pixel is passed to the pixel to the right. The filter was also implemented on the GPU by writing *CUDA* code which can be compiled and read by the *JCuda* package. The image is split into blocks. Within a block, each thread filters a pixel and share its answer to neighbouring pixels within that block.

One difficulty is that with the introduction of *CUDA* code, the ability to "compile once, run anywhere" is difficult to keep hold of. A design choice was that the user is to compile the *CUDA* code into a `.ptx` file. This is then followed by compiling the *Java* code with the `.ptx` file into a `.jar` file which can be installed as a Plugin in *ImageJ* or *Fiji*. The compiled `.jar` file can be used by *MATLAB* as well.

## Further Reading and References
* Lo, S.E. (2020). *Characterisation of Computed Tomography Noise in Projection Space with Applications to Additive Manufacturing*. PhD thesis, University of Warwick, Department of Statistics.
* Efron, B. (2004). Large-scale simultaneous hypothesis testing: The choice of a null hypothesis. *Journal of the American Statistical Association*, 99(465):96.
* Griffin, L. D. (2000). Mean, median and mode filtering of images. *Proceedings of the Royal Society of London A: Mathematical, Physical and Engineering Sciences*, 456(2004):2995–3004.
* Charles, D. and Davies, E. R. (2003). Properties of the mode filter when applied to colour images. *International Conference on Visual Information Engineering VIE 2003*, pp. 101-104.
