# Acoustic Non-Line-of-Sight Imaging (CVPR 2019)

This repo contains the code for acoustic non-line-of-sight imaging. Our project webpage is located [here](http://www.computationalimaging.org/publications/acoustic-non-line-of-sight-imaging/).

The datasets can be downloaded at this link:  
[download datasets](https://drive.google.com/a/stanford.edu/file/d/1pnRiD3e4EQiu-akvHkCMOyghwbVtQuvt/view?usp=sharing)

Place the downloaded datasets into the 'data' folder.

The initial reconstructions can be run with the following command
```sh
$ python3 AcousticNLOSReconstruction.py all
```
Or, replace 'all' with the particular scene you wish to reconstruct. 

Then, calculate PSF for deconvolution by running
```sh
$ python3 FitGaussianPSF.py
```
The iterative reconstructions can then be computed with
```sh
$ python3 ADMMReconstruction.py all
```
