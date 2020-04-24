# oxwasp_phd

* MIT License - all source code
* Copyright (c) - `reports/`
  * See `reports/thesis/LICENSE` for more details about the thesis
* Copyright (c) 2019 Sherman Lo

Contains code to reproduce the thesis Lo, S.E. (2020). *Characterisation of Computed Tomography Noise in Projection Space with Applications to Additive Manufacturing*. PhD thesis, University of Warwick, Department of Statistics

The *Java* source code for the empirical null filter *ImageJ* plugin is in the directory `java/src` and the *Maven* project file is in `java/pom.xml`.

*MATLAB* R2019a was used.

## Abstract
X-ray computed tomography can be used for defect detection in additive manufacturing. Typically, several x-ray projections of the product at hundreds of angles are used to reconstruct the object in 3D to look for any defects. The process can be time-consuming. This thesis aims to investigate if it is possible to conduct defect detection from a single projection to speed up the process. An additive manufacturing test sample was created with voids to see if they can be detected.

The uncertainty of the projection was modelled using a compound Poisson distribution. This arises from x-ray photon arrivals being a Poisson process and each photon has random energy. This results in a linear relationship between the mean and variance of the grey values in the projection. Fitting of the compound Poisson distribution using the expectation-maximisation algorithm was unsuccessful due to identifiability issues with the model. Instead, a gamma-distributed generalised linear model was fitted onto sample variance-mean data and used for variance prediction to quantify the uncertainty.

Software, called \emph{aRTist}, was used to simulate the projection and compared with the experimental projection in the face of uncertainty by treating each pixel as a hypothesis test. To overcome the imperfections of the simulation, the empirical null filter was used to cater for model misspecification so that sensible inference was achieved. This was done by locally normalising the test statistics using the mode. Voids with diameters in the order of millimetres were detectable.

This thesis is a contribution to real-time quality control in additive manufacturing.

## How to Use (Linux recommended)
Requires *Maven*. Run the command
```
mvn -f java package
```
to compile the *Java* code. The compiled `.jar` file is stored in `java/target/Empirical_Null_Filter-1.0.jar` and can be used as an *ImageJ* plugin. Copies of libraries are stored in `java/target/libs/`. This above steps can be done using *Eclipse* instead.

Download the data from *Figshare* at https://figshare.com/s/d7371af48d950eeec592 and unzip the zip file. The data is stored in the directory `data/`. The `data/` directory can be stored elsewhere by specifying where it is in the *MATLAB* script `getDataDirectory.m`.

On startup, *MATLAB* reads in the `.jar` files from the file `javaclasspath.txt`.

The following *MATLAB* products are required:
* Curve Fitting Toolbox
* Image Processing Toolbox
* Optimization Toolbox
* Statistics and Machine Learning Toolbox

Run the file `make.m` to create the figures for the paper, this is an overnight job. 16GB of RAM recommended.

The *LaTeX* file is stored in `reports/thesis/book.tex`.
