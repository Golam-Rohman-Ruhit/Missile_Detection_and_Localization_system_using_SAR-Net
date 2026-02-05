# 🚀 Missile Streak Detection and Localization System Using SAR-Net
> **A Physics-Informed Deep Learning Approach for Real-Time Aerial Defense**

---

## Ⅰ. Project Overview
This project addresses a critical challenge in automated defense: spotting and pinpointing ballistic missiles in real-time via optical sensors. Missiles appear as dim, short-lived lines on Focal Plane Arrays, often obscured by heavy distortions from air interference, Poisson sensor noise, and background clutter.

Conventional techniques like **Maximum Likelihood Estimation (MLE)** or **GLRT** fall short due to high computational demands and performance drops under weak signal-to-noise ratios (SNR). To overcome this, we propose **SAR-Net (Spatial Attention Residual Network)**—a system that learns patterns directly from data, enabling fast and dependable solutions where speed is vital.

---

## Ⅱ. Problem Definition & Significance
### The Core Challenge
1. **Learning from Weak Signals:** Traditional modeling falters under noise. Our goal is teaching a Neural Network to distinguish streaks from clutter using learned connections rather than rigid theory-bound assumptions.
2. **Trajectory Estimation:** Predicting curved flight paths shaped by air resistance requires high flexibility, which simple straight-line methods lack.

### Research Impact
* **Unified Model:** Moves beyond sequential autoencoder-detector systems to a single-step inference model.
* **Edge Deployment:** Cutting response times to milliseconds, making compact AI models viable for hardware like **C-RAM** (Counter Rocket, Artillery, and Mortar).

---

## Ⅲ. Neural Architecture (SAR-Net)
SAR-Net integrates a unique **Spatial Attention Module (SAM)** into residual blocks to prioritize meaningful features over background interference.

### Architecture Breakdown:
* **Residual Feature Extraction:** Built using ResBlocks to enable deeper layers without losing signal strength.
* **Spatial Attention Mechanism:** Unlike standard CNNs, SAM identifies the "streak" and downweights irrelevant noise before making decisions.
* **Dual-Head Prediction:**
    * **Classification Head:** Generates likelihood via Sigmoid activation.
    * **Regression Head:** Forecasts smooth position vectors for the trajectory.

> [!NOTE]
> **Architecture Diagram **

> <img width="410" height="1422" alt="Image" src="https://github.com/user-attachments/assets/48f3a5c0-6d1d-45c2-a676-0b17c8dd7672" />


> *Figure 1: Proposed SAR-Net architecture featuring Residual Blocks and SAM.*

---

## Ⅳ. Proposed Methodology
### 🧪 Physics-Informed Data Simulation
Since actual missile data is restricted, we utilize a simulator grounded in physical laws:
* **Non-Linear Trajectory:** **Bezier Curves** capture air resistance and gravity effects.
* **Realistic Noise:** Gaussian PSFs for the main signal and Poisson layers for atmospheric scattering.

### ⚙️ Training Strategy
* **Multi-Objective Loss:** Minimizes **Binary Cross-Entropy** (Detection) and **Mean Squared Error** (Localization).
* **Optimization:** Adam optimizer with weighted loss (Classification 1.0, Regression 2.0).

---

## Ⅴ. Performance & Evaluation
### 📈 Comparative Metrics
SAR-Net is measured side-by-side with classic methods across 1,000 diverse test instances.
* **ROC Curves:** Demonstrates detection accuracy under varying noise conditions (2 dB to 15 dB).
* **Visual Verification:** Overlaying red predictions on green ground truth to verify precision.

| Feature | Conventional (GLRT/MLE) | Proposed SAR-Net |
| :--- | :--- | :--- |
| **Path Modeling** | Straight-line only | **Non-linear Bezier Curves** |
| **Speed** | Slow (Sequential scans) | **Real-time (Milliseconds)** |
| **Low SNR** | High failure rate | **Attention-driven robustness** |

<img width="691" height="205" alt="Image" src="https://github.com/user-attachments/assets/71f759ae-cf31-4cfe-a992-4941f56b2fdb" />

---

## Ⅵ. Anticipated Challenges (Ablation Study)
* **Sim-to-Real Gap:** Addressed through heavy augmentation and Poisson-Gaussian noise blends.
* **Complexity:** Navigating the loss surface for non-linear manifolds.
* **Hardware Efficiency:** Optimized for NVIDIA T4/RTX GPUs to maintain <10ms latency.

---

## Ⅶ. Project Documentation
> [!IMPORTANT]
> ### 📂 Technical Report Archive
> **Main Research Document:** [📄 Missile Streak Detection and Localization System U.docx](Missile%20Streak%20Detection%20and%20Localization%20System%20U.docx)
>
> **Section Highlights:**
> * **2.4 Regression Complexity:** Details on navigating non-linear loss surfaces.
> * **3.7 Ablation Study:** Proof of SAM module contribution in reducing interference.
> * **3.8 Monte Carlo Simulation:** Statistical validation strategy.

---

## 📜 Selected References
* [1] Balci & Tekalp, "Machine learning for missile streak detection," *ICIP*, 2021.
* [3] Zhang et al., "Attention-guided pyramid context networks," *IEEE TAES*, 2021.
* [6] K. He et al., "Deep residual learning for image recognition" (ResNet), *CVPR*, 2016.
