<div align="center">

# 🫁 Deep Learning — Pneumonia Detection from Chest X-rays

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Deep Learning](https://img.shields.io/badge/Deep%20Learning-CNN-green.svg)](https://github.com/MariaNakhle/DEEP-LEARNING---Final-Project)

**Author:** [MariaNakhle](https://github.com/MariaNakhle)  
**Repository:** [DEEP-LEARNING---Final-Project](https://github.com/MariaNakhle/DEEP-LEARNING---Final-Project)

*Advanced CNN architectures for automated pneumonia detection using chest X-ray images*

</div>

---

## 📋 Table of Contents

- [🎯 Project Overview](#-project-overview)
- [🖼️ Visual Results](#️-visual-results)
- [🏗️ Repository Structure](#️-repository-structure)
- [📂 Dataset Layout](#-dataset-layout)
- [⚙️ Requirements & Installation](#️-requirements--installation)
- [🚀 Quick Start Guide](#-quick-start-guide)
- [📊 Results & Outputs](#-results--outputs)
- [🔬 Implementation Details](#-implementation-details)
- [🛠️ Troubleshooting](#️-troubleshooting)
- [📄 License](#-license)
- [📚 Documentation](#-documentation)

---

## 🎯 Project Overview

> **Detecting pneumonia from chest X-rays using state-of-the-art deep learning techniques**

This repository contains a comprehensive deep learning project that explores multiple CNN architectures for automated pneumonia detection from chest X-ray images. The project demonstrates both custom architectures and transfer learning approaches with rigorous evaluation methodologies.

### ✨ Key Features

<table>
<tr>
<td>

**🧠 Model Architectures**
- Custom CNN (4 conv blocks)
- ResNet152V2 (frozen)
- ResNet152V2 (fine-tuned)

</td>
<td>

**📈 Training & Evaluation**
- Precision-Recall analysis
- F1 threshold optimization
- Early stopping strategies

</td>
</tr>
<tr>
<td>

**⚡ Optimizer Comparison**
- SGD with/without momentum
- Adam optimizer
- RMSprop

</td>
<td>

**🔍 Multi-class Simulation**
- Binary classification (Normal/Pneumonia)
- 3-class demo (Normal/Bacterial/Viral)
- Confusion matrix analysis

</td>
</tr>
</table>

---

## 🖼️ Visual Results

### 📸 Sample Dataset Images

<div align="center">

![Pneumonia vs Normal X-ray samples](Images_for_report/PNEUMONIA_NORMAL%20.png)

*Examples of pneumonia-affected and normal chest X-ray images from the dataset*

</div>

### 📊 Task 2: Training & Precision-Recall Analysis

<div align="center">

![Task 2 Results](Images_for_report/RESULTS_FOR_2%20%20.png)

*Training history and precision-recall curves showing model performance across different thresholds*

</div>

### 🎯 Task 3: Optimizer Comparison & Early Stopping

<div align="center">

![Task 3 Results](Images_for_report/RESULTS_FOR_3%20%20.png)

*Comprehensive optimizer comparison with learning rate sweeps and early stopping analysis*

</div>

### 🔬 Task 4: Multi-class Evaluation Results

<div align="center">

![Task 4 Results A](Images_for_report/RESULTS_FOR_4_A.png)

*Multi-class training performance metrics across different experimental configurations*

![Task 4 Results B](Images_for_report/RESULTS_FOR_4_B%20%20.png)

*Confusion matrix and detailed classification report for 3-class pneumonia detection*

</div>

---

## 🏗️ Repository Structure

<details open>
<summary><b>📁 Click to expand project files</b></summary>

```
DEEP-LEARNING---Final-Project/
│
├── 📄 Task1NEW.py          # Model Architecture Definitions
│   ├── Custom CNN architecture (4 conv blocks + dense layers)
│   ├── ResNet152V2 transfer learning (frozen & fine-tuned)
│   ├── Dataset loading and preprocessing utilities
│   ├── Model visualization (diagrams & architecture tables)
│   └── Output: images/Task1/
│
├── 📄 Task2New.py          # Training & Evaluation Pipeline
│   ├── Model training with history tracking
│   ├── Precision-Recall threshold analysis (0.1-0.9)
│   ├── F1 score optimization
│   ├── Performance visualization
│   └── Output: images/Task2/
│
├── 📄 Task3New.py          # Optimizer Comparison & Tuning
│   ├── Optimizer sweep (SGD, Adam, RMSprop)
│   ├── Learning rate experimentation
│   ├── Early stopping implementation
│   ├── Best model selection & saving
│   └── Output: images/Task3/ + best_model_task3_overall.h5
│
├── 📄 Task4.py             # Multi-class Demonstration
│   ├── 3-class simulation (Normal/Bacterial/Viral)
│   ├── Extended optimizer experiments
│   ├── Confusion matrix generation
│   ├── Classification report
│   └── Output: images/
│
├── 📁 Images_for_report/   # Visual results for documentation
├── 📁 chest_xray/          # Dataset directory
├── 📁 images/              # Generated outputs and plots
├── 📄 README.md            # This file
└── 📚 Documentation Files
    ├── deep-learning project report NEW.docx
    ├── deep-learning project report NEW.pdf
    └── פרוייקט מערכות לומדות למידה עמוקה.pdf
```

</details>

### 🔑 Key Components

| File | Purpose | Key Functions |
|------|---------|--------------|
| **Task1NEW.py** | 🏗️ Architecture | `create_cnn_without_transfer_learning()`<br>`create_cnn_with_transfer_learning_frozen()`<br>`create_cnn_with_transfer_learning_finetuned()` |
| **Task2New.py** | 🎓 Training | `train_model_with_history()`<br>`evaluate_with_thresholds()`<br>`plot_precision_recall_analysis()` |
| **Task3New.py** | ⚡ Optimization | `train_with_optimizer()`<br>`train_with_early_stopping()`<br>Hyperparameter sweeps |
| **Task4.py** | 🎯 Multi-class | 3-class simulation<br>Confusion matrix<br>Classification metrics |

---

## 📂 Dataset Layout

> **⚠️ Important:** Ensure your dataset follows this exact structure

```
chest_xray/
  └── chest_xray/
      ├── train/
      │   ├── NORMAL/        # Normal chest X-rays
      │   └── PNEUMONIA/     # Pneumonia-affected X-rays
      ├── val/               # Validation set (required for Task1)
      │   ├── NORMAL/
      │   └── PNEUMONIA/
      └── test/              # Test set
          ├── NORMAL/
          └── PNEUMONIA/
```

### 📌 Configuration Notes

- All scripts use the `DATA_PATH` variable: `os.path.join("chest_xray", "chest_xray")`
- If your dataset is elsewhere, update `DATA_PATH` in each script
- Task4.py can create an internal validation split if `val/` is missing

---

## ⚙️ Requirements & Installation

### 🐍 Python Environment

**Minimum Requirements:**
- Python 3.8+
- TensorFlow 2.x

### 📦 Core Dependencies

```bash
pip install tensorflow numpy matplotlib scikit-learn
```

### 🎨 Optional (for model diagrams)

```bash
pip install pydot graphviz
```

> **Note:** Graphviz must be installed at the OS level and `dot` must be on your PATH

### 🖥️ Full Installation

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install all dependencies
pip install tensorflow numpy matplotlib scikit-learn pydot graphviz

# For GPU support (optional)
# Ensure CUDA and cuDNN are properly configured
```

### 💻 GPU Configuration

<table>
<tr>
<td>

**For NVIDIA GPU:**
- Install CUDA Toolkit
- Install cuDNN
- Verify TensorFlow GPU support
- Check compatibility versions

</td>
</tr>
</table>

---

## 🚀 Quick Start Guide

### 1️⃣ Task 1 — Model Architecture & Visualization

<details>
<summary><b>🏗️ Create and visualize CNN architectures</b></summary>

```bash
python Task1NEW.py
```

**Outputs:**
- ✅ Model architecture summaries (console)
- 🖼️ Model diagrams saved to `images/Task1/`
- 📊 Architecture tables (when Graphviz available)
- 🎨 Sample dataset visualizations

</details>

### 2️⃣ Task 2 — Training & Precision-Recall Analysis

<details>
<summary><b>🎓 Train models and analyze performance</b></summary>

```bash
python Task2New.py
```

**What it does:**
- Loads all three model architectures from Task 1
- Trains for 20 epochs (configurable)
- Evaluates precision/recall across thresholds (0.1-0.9, step 0.05)
- Generates comprehensive performance visualizations

**Outputs:**
- 📈 Training history plots → `images/Task2/`
- 🎯 Precision-Recall curves → `images/Task2/`
- 📊 F1 score analysis → `images/Task2/`

</details>

### 3️⃣ Task 3 — Optimizer Comparison & Early Stopping

<details>
<summary><b>⚡ Optimize training with different optimizers</b></summary>

```bash
python Task3New.py
```

**Experiments:**
- 🔄 SGD (with and without momentum)
- 🚀 Adam optimizer
- 📊 RMSprop
- 🎯 Learning rate sweeps
- ⏹️ Early stopping strategies

**Outputs:**
- 📊 Optimizer comparison plots → `images/Task3/`
- 🏆 Best model saved → `best_model_task3_overall.h5`
- 📈 Learning curves for all configurations

</details>

### 4️⃣ Task 4 — Multi-class Demonstration

<details>
<summary><b>🔬 Simulate 3-class pneumonia detection</b></summary>

```bash
python Task4.py
```

**Features:**
- 🎯 3-class simulation (Normal, Bacterial, Viral)
- 🔄 Multiple optimizer/LR/epoch configurations
- 📊 Confusion matrix generation
- 📈 Detailed classification report

**Outputs:**
- 🖼️ Sample images → `images/`
- 📊 Confusion matrix → `images/`
- 📈 Training plots → `images/`

> **Note:** This is a simulated demonstration as the dataset is actually binary (Normal/Pneumonia)

</details>

---

## 📊 Results & Outputs

### 🗂️ Output Directory Structure

```
images/
├── Task1/              # Model architectures and diagrams
│   ├── model_diagrams/
│   ├── architecture_tables/
│   └── sample_images/
│
├── Task2/              # Training histories and PR analysis
│   ├── training_curves/
│   ├── precision_recall_plots/
│   └── f1_score_analysis/
│
├── Task3/              # Optimizer comparisons
│   ├── optimizer_comparison/
│   ├── learning_rate_sweeps/
│   └── early_stopping_analysis/
│
└── (root)              # Task 4 outputs
    ├── confusion_matrix/
    ├── classification_reports/
    └── sample_predictions/
```

### 💾 Saved Models

| Model File | Description | Created By |
|------------|-------------|------------|
| `best_model_task3_overall.h5` | Best performing model from optimizer sweep | Task3New.py |

---

## 🔬 Implementation Details

### 🎯 Key Features

<details>
<summary><b>🔄 Reproducibility</b></summary>

- Fixed random seeds (TensorFlow & NumPy: seed=42)
- Consistent batch size: 32
- Standard input size: 160×160 pixels
- Deterministic training pipelines

</details>

<details>
<summary><b>🧠 Transfer Learning Strategy</b></summary>

**Base Model:** ResNet152V2 (ImageNet weights)

**Two Approaches:**
1. **Frozen Base** — Train only the classification head
2. **Fine-tuned** — Unfreeze layers from index 540 onward for domain adaptation

</details>

<details>
<summary><b>⚙️ Hyperparameters</b></summary>

- **Learning Rate:** 1e-4 (default)
- **Batch Size:** 32
- **Image Size:** 160×160
- **Optimizer Options:** SGD, Adam, RMSprop
- **Training Strategy:** Early stopping with patience

</details>

<details>
<summary><b>📊 Visualization Features</b></summary>

- High-resolution figures (publication quality)
- Detailed annotations for reports
- Multiple plot types:
  - Training/validation curves
  - Precision-Recall curves
  - Confusion matrices
  - ROC curves
  - Learning rate schedules

</details>

### ⚠️ Important Notes

> **Task 4 Multi-class Simulation:**  
> The dataset is binary (NORMAL/PNEUMONIA). Task 4 creates a simulated 3-class scenario for demonstration purposes by splitting pneumonia into bacterial and viral categories.

---

## 🛠️ Troubleshooting

<details>
<summary><b>❌ Dataset Loading Errors</b></summary>

**Problem:** `image_dataset_from_directory` throws errors

**Solutions:**
- ✅ Verify `DATA_PATH` points to correct location
- ✅ Check all class folders contain valid images
- ✅ Remove any corrupted or non-image files
- ✅ Ensure folder structure matches expected layout

</details>

<details>
<summary><b>🖼️ Model Diagram Issues</b></summary>

**Problem:** Model diagrams not saving

**Solutions:**
- ✅ Install Graphviz at OS level (not just pip package)
- ✅ Add Graphviz `bin` directory to PATH
- ✅ Install pydot: `pip install pydot`
- ✅ Verify with: `dot -V` (should show version)

</details>

<details>
<summary><b>🎮 GPU / TensorFlow Issues</b></summary>

**Problem:** GPU not detected or training slow

**Solutions:**
- ✅ Verify CUDA installation: `nvcc --version`
- ✅ Check cuDNN compatibility with TensorFlow version
- ✅ Test GPU: `python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"`
- ✅ Reduce batch size if running out of memory
- ✅ Consider smaller image size for faster iteration

</details>

<details>
<summary><b>💾 Memory Issues</b></summary>

**Problem:** Out of memory during training

**Solutions:**
- ✅ Reduce batch size (e.g., 16 or 8)
- ✅ Use smaller image size (e.g., 128×128)
- ✅ Enable mixed precision training
- ✅ Close other GPU-intensive applications

</details>

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

### 📜 MIT License Summary

```
Copyright (c) 2024 MariaNakhle

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
```

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📚 Documentation

### 📖 Available Reports

- 📄 **English Report:** `deep-learning project report NEW.docx` / `.pdf`
- 📄 **Hebrew Report:** `פרוייקט מערכות לומדות למידה עמוקה.pdf`

### 🔍 What's Inside

Each report contains:
- Detailed methodology
- Architecture explanations
- Experimental results
- Performance comparisons
- Visualizations and plots
- Conclusions and future work

---

<div align="center">

### 🌟 Project Highlights

| Metric | Value |
|--------|-------|
| **Models Trained** | 3 CNN Architectures |
| **Optimizers Tested** | 4 Different Optimizers |
| **Evaluation Metrics** | Precision, Recall, F1, Accuracy |
| **Dataset Split** | Train / Validation / Test |
| **Transfer Learning** | ResNet152V2 (ImageNet) |

---

### 🤝 Contributing

Contributions, issues, and feature requests are welcome!  
Feel free to check the [issues page](https://github.com/MariaNakhle/DEEP-LEARNING---Final-Project/issues).

---

### 📬 Contact

**MariaNakhle** - [@MariaNakhle](https://github.com/MariaNakhle)

**Project Link:** [https://github.com/MariaNakhle/DEEP-LEARNING---Final-Project](https://github.com/MariaNakhle/DEEP-LEARNING---Final-Project)

---

### ⭐ Show Your Support

If this project helped you, please give it a ⭐️!

---

**Made with ❤️ and 🧠 Deep Learning**

</div>

