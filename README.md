# 🚀 Missile Streak Detection and Localization System Using SAR-Net
> **A Physics-Informed Deep Learning Approach for Real-Time Defense Systems**

---

## Ⅰ. Project Overview
[cite_start]This project addresses a key challenge in automated defense: spotting and pinpointing ballistic missiles in real time via optical sensors[cite: 3]. [cite_start]Missiles appear as dim, short-lived lines on Focal Plane Arrays[cite: 4]. [cite_start]Detection is difficult due to heavy distortions from air interference, Poisson-type sensor noise, and background clutter[cite: 5].

[cite_start]While conventional techniques like **Maximum Likelihood Estimation (MLE)** or **GLRT** are mathematically solid, they demand high computing power and fail when signal strength is weak or flight paths turn irregular and curved[cite: 6, 7]. [cite_start]To solve this, we propose **SAR-Net (Spatial Attention Residual Network)**—a system that learns patterns directly from data rather than relying on hand-crafted statistics[cite: 8, 9].

---

## Ⅱ. Neural Architecture & Design
[cite_start]SAR-Net is a tailored deep learning model built for detecting signals under low-SNR conditions[cite: 87]. [cite_start]It operates as a unified system handling multiple tasks simultaneously: detection and localization[cite: 88].

### 🛠️ Key Components
* [cite_start]**Residual Feature Extraction:** Uses **ResBlocks** with skip connections to maintain gradient flow, ensuring weak missile signals remain visible through deep layers[cite: 90, 91].
* [cite_start]**Spatial Attention Mechanism (SAM):** Unlike traditional layers, SAM identifies relevant "streak" regions and downweights irrelevant noise dynamically[cite: 94, 96].
* [cite_start]**Dual-Head Prediction:** The network splits into a **Classification Head** (Sigmoid for threat likelihood) and a **Regression Head** (Linear for path position prediction)[cite: 98, 100, 101].


> **Architecture Visualization:**
> ![SAR-Net Architecture](image_b02fa2.jpg)
> [cite_start]*Figure 1: Proposed SAR-Net architecture featuring Residual Blocks and SAM[cite: 102].*

---

## Ⅲ. Methodology & Simulation
### 🧪 Physics-Informed Data Simulation
[cite_start]Because actual missile images are restricted, we use a simulation approach grounded in physical laws[cite: 16, 81].
* [cite_start]**Non-Linear Trajectory Modeling:** We apply **Bezier Curves** to capture how air resistance and gravity bend a missile’s path[cite: 17, 83].
* [cite_start]**Atmospheric Noise:** We integrate random noise patterns, **Gaussian PSFs**, and **Poisson-based layers** to mimic light scattering[cite: 84].

### ⚙️ Training Strategy
* [cite_start]**Combined Loss:** We use a multi-objective setup reducing both **Binary Cross-Entropy** (for spotting threats) and **Mean Squared Error** (for predicting motion)[cite: 18, 106].
* [cite_start]**Optimization:** Trained using the **Adam optimizer** (rate 0.001) over 20 epochs with a batch size of 32[cite: 113, 115].

---

## Ⅳ. Performance & Evaluation
### 📈 Training Metrics
The model shows rapid convergence in both accuracy and loss reduction.

![Training Metrics](training_metrics.png)
[cite_start]*Figure 2: Convergence of Detection Accuracy and Total Loss over 20 epochs[cite: 132].*

### 👁️ Visual Ground Truth Verification
Qualitative tests show the model accurately targets the streak pattern (Red) compared to the actual Ground Truth (Green), even in heavy noise.

![Visual Evaluation](final_detection_output.png)
[cite_start]*Figure 3: Visual Evaluation on Test Data[cite: 136].*

### ⚖️ Research Comparison
| Feature | Balci & Tekalp [1] | Virtanen et al. [2] | **Proposed SAR-Net** |
| :--- | :--- | :--- | :--- |
| **Target Type** | [cite_start]Linear Streaks [cite: 252] | [cite_start]Linear Debris Streaks [cite: 252] | [cite_start]**Non-Linear (Bezier) Streaks** [cite: 252] |
| **Methodology** | [cite_start]Two-Stage (DAE + CNN) [cite: 252] | [cite_start]Standard CNN [cite: 252] | [cite_start]**End-to-End Attention ResNet** [cite: 252] |
| **Primary Task** | [cite_start]Detection & Localization [cite: 252] | [cite_start]Binary Classification [cite: 252] | [cite_start]**Simultaneous Class & Reg** [cite: 252] |

---

## Ⅴ. Project Documentation
> [!IMPORTANT]
> ### 📂 Detailed Research Reports
> **Main Technical Report:** [📄 Missile Streak Detection and Localization System U.docx](Missile%20Streak%20Detection%20and%20Localization%20System%20U.docx)
>
> *Key Sections within the report:*
> [cite_start]* **Section 2.1:** Data Acquisition & Domain Gap (Sim-to-Real)[cite: 56].
> [cite_start]* **Section 3.7:** Proposed Ablation Study (Impact of SAM)[cite: 151].
> [cite_start]* **Section 3.8:** Comparative Monte Carlo Analysis Strategy[cite: 161].

---

## 🛠️ Implementation Framework
* [cite_start]**Environment:** Python 3.10, TensorFlow 2.x, Keras[cite: 144].
* [cite_start]**Vision Processing:** OpenCV for Bezier path rendering and Gaussian PSF production[cite: 145].
* [cite_start]**Hardware:** Optimized for **NVIDIA T4/RTX GPUs**, achieving inference speeds of **< 10 ms per prediction**[cite: 148, 150].

---

## 📜 References
* [cite_start]**[1]** H. E. Balci and A. M. Tekalp, "Machine learning for missile streak detection and localization," *ICIP*, 2021[cite: 259].
* [cite_start]**[2]** J. Virtanen et al., "Deep learning for space debris streak detection," *AMOS*, 2019[cite: 260].
* [cite_start]**[3]** T. Zhang et al., "Attention-guided pyramid context networks," *IEEE TAES*, 2021[cite: 261].
* [cite_start]**[6]** K. He et al., "Deep residual learning for image recognition," *CVPR*, 2016[cite: 266].
