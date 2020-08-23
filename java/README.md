# Mode Filter and Empirical Null Filter

* MIT License - all source code
* Copyright (c) 2020 Sherman Lo

*ImageJ* plugins for the mode filter and empirical null filter. The mode filter is an edge preserving smoothing filter by taking the mode of the empirical density. This may have applications in image processing such as image segmentation. The filters were also implemented on a GPU using *CUDA* and *JCuda*. This speeds up the filtering by a huge margin.

Where appropriate, please cite the thesis Lo, S.E. (2020). *Characterisation of Computed Tomography Noise in Projection Space with Applications to Additive Manufacturing*. PhD thesis, University of Warwick, Department of Statistics.

<img src=.././publicImages/mandrillExample.jpg width=800><br>
The mode filter applied on the [Mandrill test image](http://sipi.usc.edu/database/database.php?volume=misc). Top left to top right, bottom left to bottom right: mandrill test image, then the mode filter with radius of 2, 4, 8, 16, 32, 64, 128 applied.

## How to Compile (Linux recommended)
Requires *Maven* as well as *Java Runtime Environment* and *Java Development Kit*. For the use of GPU, requires *CUDA developement kit* which should include a *nvcc* compiler.

Clone this repository.

Optional for GPU: go to `/java/cuda/` and run `make` to compile the *CUDA* code into a `.ptx` file.

Go to `/java/` and run

```
mvn package
```
to compile the *Java* code. The compiled `.jar` file is stored in `java/target/Empirical_Null_Filter-2.0.jar` and can be used as an *ImageJ* plugin. Copies of libraries are stored in `java/target/libs/` and would need to be installed in *ImageJ* as well.

## About the Mode Filter
The mode filter is an image filter much like the mean filter and median filter. They process each pixel in an image. For a given pixel, the value of the pixel is replaced by the mean or median over all pixels within a distance *r* away. The mean and median filter can be used in *ImageJ*, it results in a smoothing of the image.

<img src=.././publicImages/filters.jpg width=800><br>
Top left: [Mandrill test image](http://sipi.usc.edu/database/database.php?volume=misc). Top right: Mean filter with radius 32. Bottom left: Median filter with radius 32. Bottom right: Mode filter with radius 32.

The mode filter is a by-product of the empirical null filter. Instead of taking the mean or median, the mode is taken, more specifically, the argmax of the empirical density. The optimisation problem was solved using the Newton-Raphson method. Various random initial values were tried to home in on the global maximum. Because the filtered image is expected to be smooth, the different initial values were influenced by neighbouring pixels to aid in the optimisation problem.

The resulting mode filtered image gives a smoothed image which has an impasto effect and preserved edges. This may have applications in noise removal or image segmentation.

The mode filtered was implemented on the CPU by modifying existing *Java* code from *ImageJ*. Each thread filters a row of the image in parallel from left to right. The solution to one pixel is passed to the pixel to the right. The filter was also implemented on the GPU by writing *CUDA* code which can be compiled and read by the *JCuda* package. The image is split into blocks. Within a block, each thread filters a pixel and share its answer to neighbouring pixels within that block.

One difficulty is that with the introduction of *CUDA* code, the ability to "compile once, run anyway" is difficult to keep hold of. A design choice was that the user is to compile the *CUDA* code into a *.ptx* file. This is then followed by compiling the *Java* code with the *.ptx* file into a *.jar* file which can be installed as a Plugin in *ImageJ* or *Fiji*.
