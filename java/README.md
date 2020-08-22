# Mode Filter and Empirical Null Filter

* MIT License - all source code
* Copyright (c) 2020 Sherman Lo

*ImageJ* plugins for the mode filter and empirical null filter. The mode filter is an edge preserving smoothing filter by taking the mode of the empirical density. This may have applications in image processing such as image segmentation.

After the publication of the thesis, the filters were also implemented on a GPU using *CUDA* and *JCuda*. This speeds up the filtering by a huge margin.

Please cite the thesis Lo, S.E. (2020). *Characterisation of Computed Tomography Noise in Projection Space with Applications to Additive Manufacturing*. PhD thesis, University of Warwick, Department of Statistics.

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
