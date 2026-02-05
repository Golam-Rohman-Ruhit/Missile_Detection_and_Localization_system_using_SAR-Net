# 🧠 Neural Network & Deep Learning: Comprehensive Project Series
> **Exploring Architectures from Classical Statistics to Spatial Attention Residual Networks (SAR-Net)**

---

## 🛰️ Project Overview: Missile Streak Detection
[cite_start]This project addresses a critical challenge in automated defense: spotting and pinpointing ballistic missiles in real-time via optical sensors[cite: 3]. [cite_start]Missiles appear as dim, short-lived lines on Focal Plane Arrays, often distorted by atmospheric interference and Poisson sensor noise[cite: 4, 5].

---

## 🔬 Experiment: Missile Streak Detection and Localization Using SAR-Net
<img src="https://img.shields.io/badge/Status-Completed-blue?style=for-the-badge"> <img src="https://img.shields.io/badge/Model-SAR--Net-red?style=for-the-badge&logo=pytorch">

### 1.1 Core Task and Problem Definition
[cite_start]The main goal is building a reliable system to spot weak, random signals in noisy conditions where classic statistical methods like GLRT fail[cite: 28, 29].
* [cite_start]**Learning from weak signals:** Training a Neural Network to distinguish streaks from background clutter without relying on fixed, theory-bound formulas[cite: 33, 34].
* [cite_start]**Trajectory Estimation:** Modeling non-linear, curved flight paths shaped by air resistance and gravity using Bezier curves[cite: 35, 83].

### 1.2 SAR-Net Architecture
[cite_start]Instead of treating all pixels equally, this system integrates a **Spatial Attention Module (SAM)** to identify relevant streak areas and downweight noise[cite: 10, 11].

> [!NOTE]
> #### 🖼️ Model Architecture Visualization
> Place your model architecture diagram (Figure 1) here:
> ![SAR-Net Architecture](image_b02fa2.jpg)

### 1.3 Methodology & Evaluation
* [cite_start]**Physics-Informed Simulation:** Due to limited data access, we generate realistic flight paths using physics-based simulators and Gaussian PSFs[cite: 57, 58, 84].
* [cite_start]**Multi-Objective Loss:** The model reduces Binary Cross-Entropy for detection and MSE for motion prediction[cite: 18].
* [cite_start]**Performance:** SAR-Net reaches an AUC above 0.95 and cuts processing time to milliseconds[cite: 167, 25].

> [!IMPORTANT]
> ### 📂 Project Documentation
> **Add Report Doc File Here:** [Link to SAR-Net Full Report](Missile%20Streak%20Detection%20and%20Localization%20System%20U.docx)

---

## 🛠️ Implementation Framework
* [cite_start]**Software:** Python 3.10, TensorFlow 2.x, Keras, and OpenCV[cite: 144, 145].
* [cite_start]**Hardware:** Optimized for NVIDIA T4/RTX GPUs to achieve <10ms per prediction[cite: 148, 150].
* [cite_start]**Preprocessing:** Input images are scaled [0, 1] and coordinates adjusted for $64 \times 64$ frames[cite: 146, 147].

---

## 📜 Key References
* [cite_start]**[1]** Balci & Tekalp, "Machine learning for missile streak detection and localization," ICIP, 2021[cite: 259].
* [cite_start]**[3]** Zhang et al., "Attention-guided pyramid context networks for infrared small target detection," IEEE TAES, 2021[cite: 261].
* [cite_start]**[6]** K. He et al., "Deep residual learning for image recognition" (ResNet), CVPR, 2016[cite: 266].
