---
title: Brain Tumor Detection System
emoji: 🧠
colorFrom: blue
colorTo: indigo
sdk: docker
app_file: app.py
pinned: false
---
# 🧠 Brain Tumor Detection System

An AI-powered web application for detecting brain tumors from MRI scans using deep learning and explainable AI (Grad-CAM). The system not only predicts tumor type but also highlights the affected region and generates a downloadable medical-style report.

---

## 🚀 Live Demo

👉 [https://huggingface.co/spaces/Arpan2005/brain-tumor-detector](https://huggingface.co/spaces/Arpan2005/brain-tumor-detector)

---

## 📌 Features

* 🧠 Brain tumor classification from MRI images
* 🎯 High accuracy deep learning model (Xception-based)
* 🔥 Grad-CAM visualization (tumor localization)
* 📍 Tumor region detection (Left/Right + Brain Lobe)
* ⚠️ Severity estimation
* 📄 Auto-generated PDF medical report
* 🌐 Deployable API (usable with custom frontend like Vercel)

---

## 🧠 Model Selection

For the task of brain tumor classification using MRI images, an appropriate deep learning architecture was required that could effectively capture complex spatial features and patterns present in medical images.

### 🔹 Choice of Model: Transfer Learning with Xception

In this project, a **transfer learning approach** was adopted using the **Xception** model. This model is a state-of-the-art convolutional neural network (CNN) architecture pre-trained on the **ImageNet** dataset.

The Xception model was selected due to the following reasons:

* **Depthwise Separable Convolutions**
  Xception replaces standard convolutions with depthwise separable convolutions, improving efficiency and feature learning.

* **Strong Feature Extraction**
  Pre-trained on millions of images, it captures complex visual patterns effectively—even in medical imaging.

* **High Performance**
  Achieves better accuracy with fewer parameters compared to traditional CNNs.

---

### 🏗️ Model Architecture

The model combines a pre-trained base with custom classification layers:

**Base Model:**

* Xception (without top layers)
* Input size: **299 × 299 × 3**
* Weights: ImageNet

**Custom Layers:**

* Flatten
* Dropout (0.3)
* Dense (128 neurons, ReLU)
* Dropout (0.25)
* Output Layer (4 classes, Softmax)

---

### 🧾 Classification Classes

* Glioma
* Meningioma
* Pituitary Tumor
* No Tumor

---

### ⚡ Why Transfer Learning?

Medical datasets are usually limited. Transfer learning helps by:

* Reducing training time
* Improving performance with small datasets
* Preventing overfitting

---

### ⚙️ Training Strategy

* Optimizer: **Adamax**
* Loss Function: Categorical Crossentropy
* Metrics: Accuracy, Precision, Recall
* Epochs: 10

---

### 📊 Performance

* Training Accuracy: **~99.93%**
* Validation Accuracy: **~99.54%**
* Test Accuracy: **~99.24%**

---

### ✅ Conclusion

The Xception-based transfer learning approach proved highly effective for brain tumor classification, offering strong feature extraction, high accuracy, and efficient training—making it well-suited for medical imaging applications.

---

## 🔥 Explainability (Grad-CAM)

The system uses Grad-CAM to:

* Highlight tumor regions in MRI
* Provide visual explanation of predictions
* Increase trust in AI decisions

---

## 📄 PDF Report Generation

The app generates a complete medical-style report including:

* Patient details
* Prediction & confidence
* Tumor location
* Severity level
* MRI + Grad-CAM images
* Disclaimer

---

## 🛠️ Tech Stack

* TensorFlow / Keras
* Xception (Transfer Learning)
* OpenCV
* NumPy
* Gradio
* FPDF (PDF generation)
* Hugging Face Spaces (Deployment)

---

## ⚠️ Disclaimer

This system is intended for **educational and research purposes only**.
It is **not a substitute for professional medical diagnosis**. Always consult a qualified healthcare provider.

---

## 👨‍💻 Author

**Arpan Pal**
AI & ML Developer

---

## ⭐ Future Improvements

* Patient history tracking
* Doctor dashboard
* Multi-scan comparison
* Mobile-friendly UI
* Cloud database integration

---

If you want, I can also:

* make this README more **GitHub-style (badges, screenshots, etc.)**
* or convert it into a **research paper format (IEEE/APA)**
