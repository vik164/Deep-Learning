# Deep Learning: Custom CNN vs. ResNet-50 Transfer Learning

This project explores image classification on the CIFAR-10 dataset using **PyTorch**. It features a comparative study between a custom-built 3-layer Convolutional Neural Network (CNN) and a pre-trained ResNet-50 model leveraging Transfer Learning.

---

## 🛠 Features

### 1. Custom Deep Learning Pipeline
A modular and reusable training framework implemented from scratch:
* **`train_epoch`**: Handles the full training cycle—forward pass, loss calculation, backpropagation, and parameter optimization.
* **`eval_model`**: Evaluates performance using `torchmetrics`, handling accuracy and loss tracking in non-gradient mode.
* **`run_experiment`**: Orchestrates the training process over multiple epochs, logging metrics for visualization.

### 2. Model Architectures
* **SimpleCNN**: A custom architecture featuring three convolutional layers with ReLU activations and Max Pooling, designed to demonstrate the fundamentals of feature extraction.
* **ResNet50 Transfer Learning**: 
    * Utilizes the **ImageNet-V2** pre-trained backbone.
    * Implements **Weight Freezing** to preserve low-level feature detectors.
    * Features a custom **Classification Head** (Sequential MLP) optimized for the 10 classes of CIFAR-10.

---

## 📊 Experimental Results (5 Epochs)

| Model | Test Accuracy | Final Test Loss |
| :--- | :--- | :--- |
| **SimpleCNN** | ~55% | ~1.2 |
| **ResNet-50 (Transfer)** | **~81%** | **~0.5** |

### Key Takeaways:
- **Efficiency:** The Transfer Learning model achieved significantly higher accuracy in fewer epochs because it began with high-level features already learned from ImageNet.
- **Convergence:** ResNet-50 demonstrated faster loss reduction, highlighting the robustness of deeper, pre-trained residual architectures.

---

## 🚀 Getting Started

### Prerequisites
* Python 3.x
* PyTorch & torchvision
* Torchmetrics
* Matplotlib (for results plotting)

### Installation
```bash
pip install torch torchvision torchmetrics matplotlib
