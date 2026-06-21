# 🦴 Knee Osteoarthritis Detection Pipeline (DenseNet121)

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)


An end-to-end Machine Learning pipeline and web application designed to detect and classify Knee Osteoarthritis from radiographic (X-ray) images. Built as a comprehensive computer vision project, this repository demonstrates the engineering journey from handling severe class imbalances in medical datasets to deploying a clinically safe, high-recall diagnostic tool.

## Table of Contents
- [Detailed Project Overview](#detailed-project-overview)
- [The Clinical Challenge & Engineering Solutions](#the-clinical-challenge--engineering-solutions)
- [Structural Details & Technical Architecture](#structural-details--technical-architecture)
- [Results & Metrics](#results--metrics)
- [Project Structure](#project-structure)
- [Installation & Usage](#installation--usage)
- [Web Application (Streamlit)](#web-application-streamlit)

## Detailed Project Overview

Knee Osteoarthritis (OA) is a highly prevalent degenerative joint disease. Clinically, the severity of OA is graded using the **Kellgren-Lawrence (KL) grading scale**, which ranges from Grade 0 (Healthy) to Grade 4 (Severe). 

Standard multiclass classification models often struggle with the KL scale because the visual boundary between consecutive grades (e.g., Grade 1 "Doubtful" vs. Grade 2 "Minimal") is highly ambiguous, even for trained orthopedic surgeons. To maximize clinical utility, patient safety, and model reliability, this project reformulates the 5-class ordinal scale into a highly actionable **Binary Classification** problem:

* **Negative / Doubtful (Class 0):** Encompasses KL Grades 0 and 1. Patients in this category generally do not require immediate surgical intervention and are recommended for standard monitoring.
* **Positive / Definite (Class 1):** Encompasses KL Grades 2, 3, and 4. These images present definite osteophytes (bone spurs) and joint space narrowing, representing an actionable medical condition requiring treatment.

**Primary Optimization Goal:** In medical diagnostics, a False Negative (sending a sick patient home without treatment) is significantly more dangerous than a False Positive (sending a healthy patient for a follow-up). Therefore, the primary objective of this pipeline is to maximize **Recall (Sensitivity)** while maintaining a robust overall accuracy.

## The Clinical Challenge & Engineering Solutions

Medical image datasets are notoriously imbalanced. The dataset utilized for this project featured a severe 13-to-1 class imbalance between healthy knees (Grade 0) and severe cases (Grade 4). Standard training loops lead to models that memorize the majority class, achieving superficial accuracy while completely failing on rare cases.

### The Iterative Engineering Process:
1. **Initial Categorical Baseline:** A standard ResNet50 classifier was trained. It achieved ~67% accuracy but suffered from majority-class collapse, effectively ignoring severe OA cases.
2. **Ordinal Regression Implementation:** The architecture was modified to output a continuous severity spectrum rather than discrete categorical buckets. Using a custom Weighted Mean Squared Error loss, the model achieved a **95.9% clinical accuracy** (defined as exact matches + acceptable ±1 KL grade human-level variance). This proved the model successfully learned the disease's visual progression.
3. **Optimized Binary Classifier (Final Phase):** Returning to a classification objective for definitive diagnostic output, the dataset was mapped to the binary clinical groups. The architecture was upgraded, and advanced optimization techniques (Focal Loss, Label Smoothing, AdamW) were applied to heavily penalize False Negatives.

## Structural Details & Technical Architecture

The final pipeline is structurally divided into a robust data preprocessing module, a specifically chosen deep learning architecture, and a clinically tuned inference engine.

### 1. Data Processing Pipeline
* **Input Normalization:** X-ray images are standardized to 224x224 pixels. 
* **Channel Duplication:** Native grayscale images are expanded to 3 channels to align with the expected input dimensions of ImageNet pre-trained models.
* **Standardization:** Images are normalized using standard ImageNet mean `[0.485, 0.456, 0.406]` and standard deviation `[0.229, 0.224, 0.225]`.

### 2. Model Architecture: DenseNet121
The pipeline utilizes **DenseNet121** (Pre-trained on ImageNet1K_V1) rather than standard ResNet variants. 
* **Architectural Advantage:** DenseNet's dense connectivity pattern concatenates feature maps from preceding layers to subsequent ones. This extreme feature reuse is highly effective in medical imaging, as it preserves low-level morphological details (like microscopic osteophytes and subtle joint space narrowing) deep into the network.
* **Custom Classifier Head:** The default classification head is replaced with a `Dropout(p=0.5)` layer for heavy regularization, followed by a fully connected `Linear` layer mapping to 2 output features (Negative vs. Positive).

### 3. Optimization and Loss Mapping
* **Focal Loss:** Replaced standard Cross-Entropy Loss with a custom Focal Loss implementation ($\gamma=2.0$, $\alpha$ dynamically weighted). This mathematically forces the model to divert its attention away from "easy" healthy knees and focus heavily on hard, borderline Grade 1 vs. Grade 2 cases.
* **Label Smoothing:** Applied a smoothing factor of `0.1` to prevent the network from becoming overconfident on ambiguous X-rays, significantly improving generalization.
* **Optimizer:** Utilized `AdamW` with a weight decay of `1e-2` to decouple L2 regularization from gradient updates.
* **Two-Phase Training:** * *Phase 1 (Warm-up):* Base model frozen; only the classifier head trained with a higher learning rate.
  * *Phase 2 (Fine-tuning):* Entire network unfrozen and trained using a Cosine Annealing Learning Rate scheduler.

### 4. Inference Strategy
* **Decoupled Threshold Tuning:** The inference script extracts the raw Softmax probability distributions rather than relying on an argmax function. This exposes a configurable `Threshold` variable (default `0.50`), allowing clinicians to manually adjust the model's sensitivity/specificity trade-off on the fly without retraining the network.

## Results & Metrics

By combining DenseNet121 with Focal Loss, the model intentionally sacrificed a negligible amount of strict accuracy to achieve a massive boost in **Recall**—the most crucial metric for patient safety.

| Metric | Score | Note |
|--------|-------|------|
| **Overall Accuracy** | `80.62%` | Highly competitive for complex KL-grading datasets |
| **Class 1 Recall** | `80.86%` | **Minimized False Negatives** |
| **Class 1 F1-Score**| `0.7841` | Balanced precision/recall performance on positive cases |

*The final model correctly identified over 80% of actual sick patients, successfully flagging ambiguous borderline cases that standard Cross-Entropy models completely missed.*

## Project Structure

```text
├── Data/                       
│   ├── train/                  # Training images organized by KL Grade (Not included in repo)
│   ├── val/                    # Validation dataset
│   └── test/                   # Unseen testing dataset
├── Models/                     
│   └── best_densenet121_binary.pth # Pre-trained model weights (Requires downloading/placement)
├── Model Training.ipynb        # Comprehensive Jupyter Notebook containing EDA and the full training pipeline
├── inference.py                # Standalone Python script for programmatic model evaluation
├── app.py                      # Streamlit web application for interactive UI-based inference
├── README.md                   # Project documentation and architectural overview
└── requirements.txt            # Python environment dependencies
```



## Installation & Usage
1. Clone the Repository
```Bash
git clone https://github.com/Afzaal-Ahmad-Durrani/Knee-Osteoarthritis-Detection.git

cd Knee-Osteoarthritis-Detection
```
2. Install Dependencies
Ensure you have Python 3.8+ installed.


```Bash
pip install -r requirements.txt
```

3. Run the Jupyter Notebook (Training)
If you wish to examine the training pipeline, review the confusion matrices, or retrain the model locally:

```Bash
jupyter notebook "Model Training.ipynb"
```
Web Application (Streamlit)
This repository includes a user-friendly Streamlit web interface that allows end-users to upload a radiographic image and receive a real-time AI diagnosis. The application features an interactive Diagnostic Threshold slider, explicitly demonstrating how adjusting decision boundaries impacts medical diagnostics.

Launching the Application:
Ensure your trained weights file (best_densenet121_binary.pth) is placed inside the ./Models/ directory.

Execute the following command in your terminal:

```Bash
streamlit run app.py
```
The application will initialize and open automatically in your default web browser (typically accessible at http://localhost:8501).

