# Deepfake Audio and Adversarial Attack Detection using Hybrid Deep Learning Models

## 🧠 Overview
Deepfake technology utilizes artificial intelligence to generate fake audio or video content that appears convincingly real. This poses a significant risk to privacy, trust systems, and digital security. Adversarial attacks, such as the Fast Gradient Sign Method (FGSM), subtly manipulate input data to deceive deep learning models, leading to erroneous predictions.
This project explores and enhances deepfake audio detection through a hybrid Transformer + CNN model, which is trained to recognize both original and manipulated audio—even under adversarial conditions.

## 🛡️ Objective
To strengthen digital audio verification systems through adversarial robustness, ensuring reliable protection against audio-based impersonation and manipulation attacks.

## 🔬 Research Highlights
- **Adversarial Attacks:** Implementation and testing of FGSM and white-box attacks to assess model robustness.
- **Dataset Utilization:**
  - 6,672 original and fake audio samples for initial testing.
  - 52,982 samples for comprehensive model training and evaluation.
- **Model Accuracy:**
  - CNN: 99.08%
  - CRNN: 96.27%
  - **Transformer+CNN (Proposed): 99.89%**

## 🚀 Proposed System
Our enhanced audio fraud detection system combines:
- **Spectrogram Analysis**
- **Adversarial Training Techniques**
- **Hybrid Transformer + CNN model**

## 📊 Major Contributions

✔️ Use of a **hybrid deep learning model** (Transformer + CNN) for higher accuracy  
✔️ Integration of a **User Interface (UI)** for better accessibility  
✔️ Training with **larger and diverse datasets**  
✔️ Application of the **Fake-or-Real (FoR) dataset merging strategy** for better generalization

**Dataset :** https://www.kaggle.com/datasets/mohammedabdeldayem/the-fake-or-real-dataset

**Base Paper :** https://academia.kaust.edu.sa/en/publications/audio-deepfake-detection-adversarial-attacks-and-countermeasures

