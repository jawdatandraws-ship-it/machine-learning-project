# Hybrid Deep Learning & Machine Learning for Handwritten Digit Recognition

This repository contains my university project for **ACIT4830 – Machine Learning** at **OsloMet**.

📄 Full report: *Hybrid Deep Learning and Machine Learning Approaches for Handwritten Digit Recognition*

---

## Project Overview
Handwritten digit recognition is a classic computer vision problem used in robotics, automation, and intelligent control systems.

This project compares two approaches for classifying handwritten digits using the **MNIST dataset**:

1️⃣ **Pure Deep Learning Model**
- Convolutional Neural Network (CNN)

2️⃣ **Hybrid Deep Learning + Machine Learning Model**
- CNN feature extractor + Support Vector Machine (SVM)

The goal was to investigate whether combining deep learning with traditional machine learning improves performance.

---

## Dataset
The project uses the **MNIST dataset**:
- 70,000 grayscale images of handwritten digits
- 60,000 training images
- 10,000 testing images
- Image size: 28 × 28 pixels
- 10 classes (digits 0–9)

---

## Technologies Used
- Python
- TensorFlow / Keras
- Scikit-learn
- NumPy & Pandas
- Matplotlib & Seaborn
- Google Colab

---

## Methodology

### Data Preprocessing
- Images normalized to range [0,1]
- Dataset split:
  - 80% Training
  - 20% Validation
  - Separate Test set

---

## Model 1 — Convolutional Neural Network (CNN)

Architecture:
- 2 Convolutional blocks
- Batch Normalization
- Max Pooling
- Dropout Regularization
- Dense layers with Softmax output
- Optimizer: **Adam**
- EarlyStopping to prevent overfitting

---

## Model 2 — Hybrid CNN + SVM

Pipeline:
1. CNN trained normally
2. Extract **128-dimensional feature vectors**
3. Train **SVM (RBF kernel)** on CNN features
4. Compare performance with pure CNN

This hybrid approach combines:
- CNN → feature extraction power  
- SVM → strong decision boundaries

---

## Results

| Model | Test Accuracy | ROC AUC |
|------|---------------|---------|
| CNN | **98.91%** | 0.9999 |
| CNN + SVM | **98.93%** | 0.9999 |

Key findings:
- Both models achieved **>98% accuracy**
- Hybrid CNN+SVM slightly outperformed CNN
- ROC curves show near-perfect classification
- Confusion matrices show very few misclassifications

---

## Key Insights
- CNNs learn strong visual features automatically
- SVM improves classification boundary precision
- Hybrid models can slightly improve performance
- Approach is suitable for **robot vision systems**

---

## Future Work
Possible improvements:
- Use more complex datasets (EMNIST, real-world images)
- Apply transfer learning (ResNet, VGG, MobileNet)
- Deploy on edge devices (Raspberry Pi / Jetson)
- Integrate into robotics vision systems

---

## Author
**Jawdat Androus**  
Oslo Metropolitan University  
ACIT4830 – Spring 2025
