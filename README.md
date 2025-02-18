# LiDAR-Camera Extrinsic Calibration
This repository provides a project about deep-learning-based targetless extrinsic calibration for LiDAR--Camera system.

## Table of Contents
- [About the Project](#about-the-project)
    * [Problem Formulation](#problem-formulation)
    * [Data Preprocessing](#data-preprocessing)
    * [Model Architecture](#model-architecture)
- [Getting Started](#getting-started)
    * [Requirements](#requirements)
    * [Training](#training)
    * [Evaluation](#evaluation)
- [Current Result](#current-result)
- [Contacts](#contacts)
- [Acknowledgments](#acknowledgments)

## About the Project 
This project was part of the sensor fusion and computer fusion research project at [WiComAI Lab](https://wireless.kookmin.ac.kr/) of Kookmin University, Seoul, South Korea. 
The project aims to predicts the misalignment in the extrinsic parameters between the camera and LiDAR by leveraging [EfficientNetV2](https://arxiv.org/abs/2104.00298) 
as feature extraction network and CSA transformer from [Lite Vision Transformer](https://arxiv.org/abs/2112.10809) as feature matching network. The deep learning network is trained 
and tested using the [KITTI Odometry dataset](https://www.cvlibs.net/datasets/kitti/eval_odometry.php).

#### Problem Formulation
Extrinsic parameters in LiDAR-Camera system is used to perform coordinate transformation of 3D poinnts from LiDAR coordinate system to camera coordinate system.
This coordinate transformation is used to perform data fusion between the two modalities:

Given a LiDAR point $P_C = \begin{bmatrix} X_C & Y_C & Z_C \end{bmatrix}^T$  $P_L: (X_L, Y_L, Z_L)$ and LiDAR point in camera coordinate system $P_L: (X_C, Y_C, Z_C)$, the coordinate transformation of a single LiDAR point
is expressed as:

$$
P_C = \begin{bmatrix} R & | & t \end{bmatrix} \cdot P_L = T \cdot P_L,
$$

where $T$, $R$, and $t$ are the extrinsic parameter matrix, rotation matrix, and translation vector of the LiDAR
camera system, respectively.

The network aims to correct the deviation in extrinsic parameters accumulated during vehicles operations, where this deviation can be expressed as:

$$ 
T_{actual} =  \Delta T \cdot T_{known}, 
$$

where $T_{actual}$ is the actual extrinsic parameter after deviation is considered, $T_{known}$ us the initial extrinsic parameter without the deviation considered, 
and $\Delta T$ is the extrinsic parameter deviation represented in a rigid transformation. 
The network predicts the value of $\Delta T$ to obtain the calibrated extrinsic parameter, which expressed as:

$$
T_{actual} = \Delta T_{predicted}^{-1} \cdot \Delta T \cdot T_{known}.
$$

Assuming that the prediction is highly accurate, the deviation term will cancel out.

#### Data Preprocessing
The network utilizes RGB images from the camera and depth images of 2D projections of LiDAR point clouds. 
The depth image is created by performing rigid transformation on point clouds from the LiDAR coordinate system to the camera coordinate 
system using the extrinsic parameters of the LiDAR-camera system. After the rigid transformation,
the point clouds is projected onto the 2D image using the intrinsic parameters of the camera, 
where each projected pixed of a point is given a pixel value corresponding to the point distance to the camera.




#### Model Architecture


## Getting Started
### Requirements


### Training


### Evaluation


## Current Result


## Contacts
Miftahul Umam

Email:
miftahul.umam14@gmail.com

## Acknowledgments