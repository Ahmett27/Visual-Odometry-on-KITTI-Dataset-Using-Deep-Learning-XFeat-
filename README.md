# XFeat Monocular Visual Odometry on KITTI Dataset

This repository contains a complete pipeline for Monocular Visual Odometry (VO) using the **XFeat (Accelerated Features)** deep learning model. The algorithm extracts robust features, matches them across consecutive frames, and estimates the camera trajectory using the **KITTI Odometry Dataset**.

## 🚀 Features
* **Deep Feature Extraction:** Uses the lightweight and highly accurate XFeat model via PyTorch Hub.
* **Pose Estimation:** Recovers camera poses using Essential Matrix estimation and RANSAC.
* **Absolute Scale Integration:** Extracts absolute scale from Ground Truth data to solve the monocular scale ambiguity.
* **Real-time Visualization:** Generates real-time video outputs for both feature matching and trajectory drawing.
* **Evaluation:** Calculates the Root Mean Square Error (RMSE) against the ground truth and outputs a comparative 2D trajectory plot.

## 🛠️ Prerequisites
Ensure you have Python installed along with the following dependencies. A CUDA-enabled GPU is highly recommended for faster XFeat inference.

```bash
pip install torch torchvision opencv-python numpy matplotlib
