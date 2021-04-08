# oxwasp_phd

* MIT License - all source code
* Copyright (c) - `reports/`
  * See `reports/thesis/LICENSE` for more details about the thesis only
* Copyright (c) 2020 Sherman Lo

Contains code to reproduce the thesis Lo, S.E. (2020). *Characterisation of Computed Tomography Noise in Projection Space with Applications to Additive Manufacturing*. PhD thesis, University of Warwick, Department of Statistics

The *Java* source code for the mode and empirical null filter *ImageJ* plugin is in the submodule `modefilter`. Please see the <a href="https://github.com/shermanlo77/modefilter">`modefilter` repository</a> for further information about the empirical null filter and mode filter *ImageJ* plugin.

*MATLAB* R2019a was used.

## Abstract
X-ray computed tomography can be used for defect detection in additive manufacturing. Typically, several x-ray projections of the product at hundreds of angles are used to reconstruct the object in 3D to look for any defects. The process can be time-consuming. This thesis aims to investigate if it is possible to conduct defect detection from a single projection to speed up the process. An additive manufacturing test sample was created with voids to see if they can be detected.

The uncertainty of the projection was modelled using a compound Poisson distribution. This arises from x-ray photon arrivals being a Poisson process and each photon has random energy. This results in a linear relationship between the mean and variance of the grey values in the projection. Fitting of the compound Poisson distribution using the expectation-maximisation algorithm was unsuccessful due to identifiability issues with the model. Instead, a gamma-distributed generalised linear model was fitted onto sample variance-mean data and used for variance prediction to quantify the uncertainty.

Software, called *aRTist*, was used to simulate the projection and compared with the experimental projection in the face of uncertainty by treating each pixel as a hypothesis test. To overcome the imperfections of the simulation, the empirical null filter was used to cater for model misspecification so that sensible inference was achieved. This was done by locally normalising the test statistics using the mode. Voids with diameters in the order of millimetres were detectable.

This thesis is a contribution to real-time quality control in additive manufacturing.

<img src=./publicImages/frontCover.jpg width=400><br>
The left-hand side shows an x-ray projection of an additive manufactured cuboid. Its edges appeared curved due to spot and panel effects and this can be fixed using shading correction. The right-hand side shows the *p*-values of the resulting inference. Lighter colours show evidence of a defect and they successfully highlighted voids put in there purposefully.

## How to Use (Linux recommended)
Requires *Maven* as well as *Java Runtime Environment* and *Java Development Kit*.

Run the command
```
mvn -f modefilter package
```
to compile the *Java* code for the empirical null filter. The compiled `.jar` files are `modefilter/target/Empirical_Null_Filter-X.X.X.jar` and other required libraries in `modefilter/target/libs/`. On startup, *MATLAB* read and recognise these `.jar` files using the file paths specified in `javaclasspath.txt`.

Download the data from [*Figshare*](https://figshare.com/s/d7371af48d950eeec592) and unzip the zip file. The data is to be stored in the directory `data/`. The location of the `data` directory is specified in the *MATLAB* script `getDataDirectory.m`.

The following *MATLAB* products are required:
* Curve Fitting Toolbox
* Image Processing Toolbox
* Optimization Toolbox
* Statistics and Machine Learning Toolbox

Run the file `make.m` to create the figures for the thesis, this is an overnight job. 16GB of RAM recommended.

The *LaTeX* file is stored in `reports/thesis/book.tex`.

## *ImageJ* plugin

Please see the <a href="https://github.com/shermanlo77/modefilter">`modefilter` repository</a> for further information about the empirical null filter and mode filter *ImageJ* plugin.


## Mode and Empirical Null Filter *MATLAB* Wrapper

The repository contains *MATLAB* wrapper code so that the mode and empirical null filter can be used in *MATLAB* for both the CPU and GPU version.

Requires *Maven* as well as *Java Runtime Environment* and *Java Development Kit*. For the use of a *Nvidia* GPU, requires *CUDA Development Kit* which should include a *nvcc* compiler.

Follow the instructions in the <a href="https://github.com/shermanlo77/modefilter">`modefilter` repository</a> for compiling the code into `.jar` files. On startup, *MATLAB* read and recognise these `.jar` files using the file paths specified in `javaclasspath.txt`.

The wrapper code are `inference/EmpiricalNullFilter.m` and `inference/EmpiricalNullFilterGpu.m`.
