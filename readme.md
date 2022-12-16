# general_ups_python
This code is a reimplementation of the following [paper](https://vision.in.tum.de/_media/spezial/bib/haefner2019iccv.pdf) in Python using Pytorch:

> **Variational Uncalibrated Photometric Stereo under General Lighting**
> *Haefner, B., Ye, Z., Gao, M., Wu, T., Quéau, Y. and Cremers, D.; In International Conference on Computer Vision (ICCV), 2019.*
> ![alt tag](https://vision.in.tum.de/_media/spezial/bib/haefner2019iccv.png)

We propose an efficient principled variational approach to uncalibrated PS under general illumination. To this end, the Lambertian reflectance model is approximated through a spherical harmonic expansion, which preserves the spatial invariance of the lighting. The joint recovery of shape, reflectance and illumination is then formulated as a single variational problem.

## 0. MATLAB version
Original MATLAB version is available [here](https://github.com/zhenzhangye/general_ups)

## 1. Requirements and Setup

### 1.1. CPU run
Has been tested under Ubuntu 20.04.5 LTS (Focal Fossa) with an Intel(R) Core(TM) i7-4702HQ CPU @ 2.20GHz, 15Gb of RAM

### 1.2. GPU run
Has been tested under Ubuntu 20.04.5 LTS (Focal Fossa) with an Intel(R) Xeon(R) CPU E5-2637 v3 @ 3.50GHz, 31Gb of RAM, and an NVIDIA GeForce GTX 1070 with 8192 MiB

### 1.3. Python libraries
Have a look at `environment.yml` to see the python dependencies, e.g. pytorch is needed, but not necessarily with GPU support.

### 1.4. Setup
```
$ conda env create -f environment.yml
$ conda activate general_ups
```
### 1.5 Download example data set
Run
```
$ bash data/download.sh
```
to download two data sets the data folder:
1) `synthetic_joyfulyell_hippie` of 32Mb
2) `xtion_backpack_sf4_ups` of  359Mb

## 2. Usage
For each of the three usages described here, we provide 
1) two example data sets, and
2) the possibility to run you own dataset
### 2.1. GeneralUPS solver with our balloon-like depth initialzation
#### 2.3.1 Run provided example(s)
```
$ python main.py example1 # Run the synthetic example data set
```
or
```
$ python main.py example2 # Run the real-world example data set 
```
#### 2.3.2 Run your own data set
Run 
1) `$ python main.py -h` and
2) `$ python main.py manual -h`

to see all possible options.

For example you can run the provided data set manually (internally this is what happens):
```
$ python main.py \
--gpu \
--output=./output/synthetic_joyfulyell_hippie \
manual \
--volume=24.77 \
--mask=data/synthetic_joyfulyell_hippie/mask.png \
--images=data/synthetic_joyfulyell_hippie/images.pth \
--intrinsics=data/synthetic_joyfulyell_hippie/K.pth \
--gt_depth=data/synthetic_joyfulyell_hippie/z_gt.pth \
--gt_light=data/synthetic_joyfulyell_hippie/l_gt_25x9x3.pth \
--gt_albedo=data/synthetic_joyfulyell_hippie/rho_gt.pth
```
Note that the order of the arguments matter, i.e. `-g`, `--gpu_id`, and `--output` has to be stated before the  `manual` keyword. 

This should result in the following error metrics:

```
rmse albedo: 0.06496411561965942
rmse_s: 0.11562050133943558
ae_s: 28.797739028930664
ae_n: 7.723351955413818 # paper reported 7.49
rmse_z: 0.6139106154441833
```
### 2.2 Other Possibilites
You can run the ballooning and the general_ups code separately as well, have a look at 
```
$ python ballooning.py -h
$ python ballooning.py manual -h
$ python general_ups.py -h
$ python general_ups.py manual -h
```
This is for example handy if 

1) you'd like to test **our** initiatlization on **your** UPS solver, or 
2) if you have **your own** depth initialization and you'd like to run **our** UPS solver on it.

## 3. Input Assumptions
- `images` of shape: `NxCxHxW`.
- binary `mask` of shape: `1xHxW`.
- Optional: `intrinsics` of shape: `3x3`. If not provided orthographic projection is assumed.
- Optional: `gt_depth` of shape: `HxW`
- Optional: `gt_light` of shape: `Nx{4 or 9}xC`
- Optional: `gt_albedo` of shape: `CxHxW`

## 4. License
general_ups is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License, see [here](http://creativecommons.org/licenses/by-nc-sa/4.0/), with an additional request:

If you make use of the library in any form in a scientific publication, please refer to `https://github.com/Bjoernhaefner/general_ups_python` and cite the paper

```
@inproceedings{haefner2019variational,
 title = {Variational Uncalibrated Photometric Stereo under General Lighting},
 author = {Bjoern Haefner and Zhenzhang Ye and Maolin Gao and Tao Wu and Yvain Quéau and Daniel Cremers},
 booktitle = {IEEE/CVF International Conference on Computer Vision (ICCV)},
 year = {2019},
 doi = {10.1109/ICCV.2019.00863},
}
```
