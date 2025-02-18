# LiDAR-Camera Extrinsic Calibration
This repository provides a project about deep-learning-based targetless extrinsic calibration for LiDAR--Camera system.

## Table of Contents
- [About the Project](#about-the-project)
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

Extrinsic parameters in LiDAR-Camera system is used to perform coordinate transformation of 3D poinnts from LiDAR coordinate system to camera coordinate system.
This coordinate transformation is used to perform data fusion between the two modalities:

Given a set of LiDAR points $P_L = \begin{bmatrix} X_L & Y_L & Z_L \end{bmatrix}$ and the point set in camera coordinate system 
$P_C = \begin{bmatrix} X_C & Y_C & Z_C \end{bmatrix}$, the coordinate transformation of the LiDAR point set is expressed as:

$$
P_C = \begin{bmatrix} R & | & t \end{bmatrix} \cdot P_L = T \cdot P_L,
$$

where T, R, and t are the extrinsic parameter matrix, rotation matrix, and translation vector of the LiDAR
camera system, respectively.

#### Data Preprocessing
The network utilizes RGB images from the camera and depth images of 2D projections of LiDAR point clouds. 
The depth image is created by performing rigid transformation on point clouds from the LiDAR coordinate system to the camera coordinate 
system using the extrinsic parameters of the LiDAR-camera system. 

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