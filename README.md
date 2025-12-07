# 👨‍💻 EigenFaces: Face Recognition using Principal Component Analysis (PCA)

This repository contains the complete implementation of the classic **EigenFaces** face recognition algorithm, developed as an exercise for the "Mathematics for Machine Learning" (M4ML) course. The algorithm leverages **Principal Component Analysis (PCA)** to achieve dimensionality reduction and extract the most distinct features from a set of face images, thereby modeling the **"Face Space"**.

---

## 🎯 Project Overview and Data

The main goal of this project was to model the subspace of valid face images and solve the **Face Identification** problem.

* **Dataset:** ORL Face Database (common for EigenFaces exercises).
* **Image Dimensions:** $112 \times 92$ pixels (Grayscale).
* **Size:** 40 persons (classes) with 10 images per person (Total: 400 images).

---

## 🛠️ Implementation Steps

The project follows the standard EigenFaces methodology across several key phases, emphasizing the mathematical underpinnings required for Machine Learning.

### 1. Data Preprocessing and Storage

* Conversion of the original PGM-format images into PNG-format using the OpenCV Python library.
* Storage of the face image database in three optimized **NumPy array** shapes:
    * **4D Volume:** $112 \times 92 \times 10 \times 40$ (Height x Width x NumImagesPerClass x TotalClasses).
    * **3D Volume:** $10304 \times 10 \times 40$ (Image\_vector x NumImagesPerClass x TotalClasses).
    * **2D Matrix:** $10304 \times 400$ (Image\_vector x TotalImages).

### 2. Data Partitioning (Train/Test Split)

* The dataset is split into training and test sets by controlling the number of training images per person using the parameter $t$.
    * Example: $t=0.7$ results in 7 training images and 3 test images per person.

### 3. Centering the Database (Mean Face)

* Calculation of the **Mean Face** ($\vec{m}$) using the Training set.
* Normalization of both the training and test data by subtracting the Mean Face from every image column-vector $\vec{f}$.
    * The column-vector of the face after normalization is $\vec{f}_{m}$.

### 4. Computing EigenFaces (PCA)

* The goal is to find the **eigenvectors** of the covariance matrix of the data. The EigenFaces are the eigenvectors of the covariance matrix of face images.
* **Technique:** **Singular Value Decomposition (SVD)** is applied to the centered 2D training matrix $\mathbf{A}$ ($D \times N$). 
* The $p$ eigenvectors corresponding to the largest eigenvalues/singular values are defined as the **EigenFaces**.

### 5. Projection and Recognition

* Selection of the top-$p$ EigenFaces to construct a projection matrix $\mathbf{W}^T$.
* Projection of the centered data into the $p$-dimensional **eigenface space**.
* **Identification:** A test face is identified as the training face whose projection in the $p$-space is closest.
* **Distance Metrics:** The distance matrix "Dist" is computed using **Euclidean distance** and **Cosine similarity**.
* **Evaluation:** The **Recognition Rate** is computed as the number of correctly identified Test faces divided by the total number of Test faces.

### 6. Face Reconstruction

* Faces are reconstructed from their lower-dimensional projections $\mathbf{F}_{transformed}$.
* The reconstructed face vector is $\hat{F}_{init}=F_{mean}+WF_{transformed}$.
* The **Total Reconstruction Error** for all test faces is calculated using the squared Frobenius norm.

---

## 🚀 Technologies

* **Python**
* **NumPy:** Essential for Linear Algebra operations (SVD, array manipulation).
* **OpenCV (`cv2`):** Used for image file handling.
* **Matplotlib:** Used for visualizing results (Mean Face, EigenFaces, Distance Heatmaps).

---

**© 2025 Konstantinos Kotsaras.**
