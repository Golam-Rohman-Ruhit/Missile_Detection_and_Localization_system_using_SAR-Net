

---

# <p align="center">🌠 Missile Streak Detection and Localization System 🚀</p>

<p align="center"><b>Using a Spatially-Attentive Residual Network (SAR-Net)</b></p>
<p align="center">
<img src="[https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python](https://www.google.com/search?q=https://img.shields.io/badge/Python-3.10%2B-blue%3Fstyle%3Dfor-the-badge%26logo%3Dpython)" alt="Python 3.10+">
<img src="[https://img.shields.io/badge/TensorFlow-2.x-orange?style=for-the-badge&logo=tensorflow](https://www.google.com/search?q=https://img.shields.io/badge/TensorFlow-2.x-orange%3Fstyle%3Dfor-the-badge%26logo%3Dtensorflow)" alt="TensorFlow 2.x">
<img src="[https://img.shields.io/badge/Keras-2.x-red?style=for-the-badge&logo=keras](https://www.google.com/search?q=https://img.shields.io/badge/Keras-2.x-red%3Fstyle%3Dfor-the-badge%26logo%3Dkeras)" alt="Keras 2.x">
<img src="[https://img.shields.io/badge/GPU-NVIDIA-green?style=for-the-badge&logo=nvidia](https://www.google.com/search?q=https://img.shields.io/badge/GPU-NVIDIA-green%3Fstyle%3Dfor-the-badge%26logo%3Dnvidia)" alt="NVIDIA GPU">
</p>

[cite_start]This project tackles a critical challenge in automated defense: real-time detection and precise pinpointing of ballistic missiles using advanced optical sensors[cite: 1, 3]. [cite_start]Missiles appear as faint, fleeting streaks on Focal Plane Arrays, often hidden by severe atmospheric interference, Poisson sensor noise, and background clutter[cite: 4, 5].

---

## ✨ Project Overview

[cite_start]Traditional methods like Maximum Likelihood Estimation (MLE) or Generalized Likelihood Ratio Test (GLRT) are computationally intensive and often fail under weak signal conditions or with irregular, curved flight paths[cite: 6, 7]. [cite_start]Our innovative solution, **SAR-Net (Spatial Attention Residual Network)**, learns complex patterns directly from data, moving beyond rigid, hand-crafted statistics[cite: 8, 9].

### 🌟 Key Innovations:

* **Spatial Attention Module (SAM)**: [cite_start]This unique module intelligently identifies relevant "streak" regions and dynamically suppresses irrelevant noise and interference[cite: 10, 11].
* **Deep Signal Retention**: [cite_start]Built with residual blocks (inspired by ResNet), SAR-Net allows for deeper feature extraction without losing the faint signal strength of subtle objects[cite: 13].
* **Physics-Informed Simulation**: [cite_start]We generate realistic flight paths using Bezier curves, incorporating the effects of gravity and air resistance, moving beyond simplistic straight-line models[cite: 16, 17].
* **End-to-End Real-Time System**: [cite_start]SAR-Net offers a unified solution for both threat classification and precise trajectory coordinate regression, achieving millisecond-level response times crucial for defense applications[cite: 25, 88].

---

## 🏗️ SAR-Net Architecture

[cite_start]The SAR-Net model is an elegant, end-to-end framework specifically designed for robust performance in low Signal-to-Noise Ratio (SNR) environments[cite: 87, 89].

| Component | Functionality |
| --- | --- |
| **Residual Feature Extraction** | [cite_start]Captures detailed spatial patterns while ensuring weak signals are not lost across deep layers via skip connections[cite: 91]. |
| **Spatial Attention Mechanism** | [cite_start]Generates pixel-specific weights to enhance streak visibility and actively suppress background clutter and noise[cite: 94, 96]. |
| **Dual-Head Prediction** | [cite_start]Splits into a **Classification Head** (Sigmoid activation for threat detection) and a **Regression Head** (linear activation for precise path coordinates)[cite: 100, 101]. |

<p align="center">
<img src="[https://i.imgur.com/example_arch_diagram.png](https://www.google.com/search?q=https://i.imgur.com/example_arch_diagram.png)" alt="SAR-Net Architecture Diagram" width="700"/>





<i><p align="center"><b>Figure 1:</b> The proposed SAR-Net architecture, showcasing Residual Blocks and Spatial Attention Modules for simultaneous detection and localization.</p></i>
</p>

---

## 📈 Performance & Visual Results

[cite_start]SAR-Net is trained using a sophisticated multi-objective loss function, combining Binary Cross-Entropy for accurate threat classification and Mean Squared Error (MSE) for precise trajectory prediction[cite: 18, 106].

### 📉 Training Metrics

[cite_start]The model demonstrates rapid and stable convergence, achieving near-perfect detection accuracy within just 20 training epochs[cite: 115, 132].

<p align="center">
<img src="[https://i.imgur.com/example_training_metrics.png](https://www.google.com/url?sa=E&source=gmail&q=https://i.imgur.com/example_training_metrics.png)" alt="Training Metrics Plot" width="800"/>





<i><p align="center"><b>Figure 2:</b> Training metrics showing the rapid convergence of Detection Accuracy (Left) and Total Loss (Right) over 20 epochs.</p></i>
</p>

### 🎯 Qualitative Evaluation

[cite_start]Visual overlays vividly confirm that SAR-Net accurately pinpoints and tracks the missile streak, effectively distinguishing it from severe noise and background distortions[cite: 135].

<p align="center">
<img src="[https://i.imgur.com/example_detection_output.png](https://www.google.com/search?q=https://i.imgur.com/example_detection_output.png)" alt="Visual Detection Output" width="800"/>





<i><p align="center"><b>Figure 3:</b> Visual evaluation on test data. The <b>Green</b> lines represent the actual Ground Truth trajectory, while the <b>Red</b> lines represent the SAR-Net prediction.</p></i>
</p>

---

## 🔬 Experiments & Detailed Documentation

[cite_start]To thoroughly validate SAR-Net, we conducted extensive Monte Carlo simulations, comparing its performance against classic methods like GLRT across various noise conditions[cite: 161, 165].

### 📊 Comparative Analysis with Existing Literature

| Feature | Balci & Tekalp [1] | Zhang et al. [3] | Virtanen et al. [2] | **Proposed SAR-Net (This Work)** |
| --- | --- | --- | --- | --- |
| **Target Type** | [cite_start]Linear Streaks [cite: 252] | [cite_start]Static IR Blobs [cite: 252] | [cite_start]Linear Debris [cite: 252] | [cite_start]**Non-Linear (Bezier) Streaks** [cite: 252] |
| **Methodology** | [cite_start]Two-Stage (AE + CNN) [cite: 252] | [cite_start]Attention-based Segmentation [cite: 252] | [cite_start]Standard CNN Classifier [cite: 252] | [cite_start]**End-to-End Attn-Residual Net** [cite: 252] |
| **Noise Handling** | [cite_start]Denoising Pre-processing [cite: 252] | [cite_start]Contextual Attention [cite: 252] | [cite_start]CNN Feature Learning [cite: 252] | [cite_start]**Learned Spatial Attention** [cite: 252] |
| **Primary Task** | [cite_start]Detection & Localization [cite: 252] | [cite_start]Semantic Segmentation [cite: 252] | [cite_start]Binary Classification [cite: 252] | [cite_start]**Simultaneous Class. & Regression** [cite: 252] |

### 🧪 Experiment 1: SNR Robustness Study

[cite_start]This experiment provides a detailed analysis of SAR-Net's detection accuracy and AUC performance across varying noise intensities, from 2 dB to 15 dB, showcasing its resilience in challenging low-SNR environments[cite: 165].

> **[INSERT LINK TO EXPERIMENT 1 REPORT (.DOC) HERE]**

---

### 🔬 Experiment 2: Ablation Study of Spatial Attention

[cite_start]A crucial study quantifying the specific performance gains contributed by the Spatial Attention Module. This involves comparing the full SAR-Net model against a baseline ResNet without the attention mechanism[cite: 151, 157].

> **[INSERT LINK TO EXPERIMENT 2 REPORT (.DOC) HERE]**

---

### 📈 Experiment 3: Non-Linear Trajectory Regression Accuracy

[cite_start]This experiment evaluates SAR-Net's ability to precisely estimate complex, curved flight paths generated via Bezier curves, demonstrating its superiority over traditional linear trajectory assumptions[cite: 83, 126].

> **[INSERT LINK TO EXPERIMENT 3 REPORT (.DOC) HERE]**

---

## 🛠️ Implementation & Environment

* **Language**: [cite_start]Python 3.10+[cite: 144].
* **Deep Learning Frameworks**: [cite_start]TensorFlow 2.x and Keras[cite: 144].
* **Computer Vision**: [cite_start]OpenCV for physics-based path and Gaussian PSF rendering[cite: 145].
* **Hardware Acceleration**: [cite_start]Optimized for NVIDIA T4/RTX GPUs, achieving sub-10ms inference times crucial for real-time defense applications[cite: 148, 150].
* **Data Preprocessing**: [cite_start]All input sensor images are scaled to [0, 1], and ground truth path coordinates are normalized to the  image resolution for consistent output[cite: 146, 147].

---

## 📚 References

* [1] H. E. Balci and A. M. Tekalp, "Machine learning for missile streak detection and localization," in *2021 IEEE International Conference on Image Processing (ICIP)*, IEEE, 2021, pp. 1659–1663.
* [2] J. Virtanen, J. Poikonen, M. Sanguino, and M. Komarik, "Deep learning for space debris streak detection in optical images," in *Proceedings of the Advanced Maui Optical and Space Surveillance Technologies Conference (AMOS)*, 2019.
* [3] T. Zhang, H. Li, L. Li, L. Wang, and T. Xu, "Attention-guided pyramid context networks for infrared small target detection," *IEEE Transactions on Aerospace and Electronic Systems*, vol. 57, no. 4, pp. 2508–2519, 2021.
* [4] Y. Dai, Y. Wu, F. Zhou, and K. Barnard, "Asymmetric contextual modulation for infrared small target detection," in *Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)*, 2021, pp. 950–959.
* [5] X. Ying, G. Liu, and Y. Liu, "Rethinking the saliency of infrared small target detection: A deep learning approach," *Neurocomputing*, vol. 396, pp. 223–234, 2020.
* [6] K. He, X. Zhang, S. Ren, and J. Sun, "Deep residual learning for image recognition," in *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2016, pp. 770–778.
* [7] K. Simonyan and A. Zisserman, "Very deep convolutional networks for large-scale image recognition," in *International Conference on Learning Representations (ICLR)*, 2015.
* [8] Y. LeCun, Y. Bengio, and G. Hinton, "Deep learning," *Nature*, vol. 521, no. 7553, pp. 436–444, 2015.
* [9] S. Ioffe and C. Szegedy, "Batch normalization: Accelerating deep network training by reducing internal covariate shift," in *International Conference on Machine Learning (ICML)*, 2015, pp. 448–456.

---
